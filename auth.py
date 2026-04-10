import os
import hmac
import base64
import hashlib
import re
from datetime import datetime, timezone, timedelta

import psycopg2


PBKDF2_ITERATIONS = 240000
MAX_LOGIN_ATTEMPTS = int(os.getenv("APP_MAX_LOGIN_ATTEMPTS", "5"))
LOCKOUT_MINUTES = int(os.getenv("APP_LOCKOUT_MINUTES", "15"))
SESSION_MAX_AGE_HOURS = int(os.getenv("APP_SESSION_MAX_AGE_HOURS", "12"))


def connect_postgres():
    """
    Single entry point for Postgres connections.
    Set POSTGRES_SSLMODE=require (or verify-full) for Neon, Supabase, Azure PG, etc.
    """
    port = os.getenv("POSTGRES_PORT") or "5432"
    kwargs = {
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("POSTGRES_HOST"),
        "port": port,
    }
    sslmode = (os.getenv("POSTGRES_SSLMODE") or "").strip()
    if sslmode:
        kwargs["sslmode"] = sslmode
    return psycopg2.connect(**kwargs)


def _get_conn():
    return connect_postgres()


def _utc_now():
    return datetime.now(timezone.utc)


def _get_session_secret() -> str:
    return (
        os.getenv("APP_SESSION_SECRET")
        or os.getenv("AZURE_OPENAI_KEY")
        or os.getenv("POSTGRES_PASSWORD")
        or "dev-session-secret"
    )


def _sign_payload(payload: str) -> str:
    secret = _get_session_secret().encode("utf-8")
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _b64url_encode(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("utf-8").rstrip("=")


def _b64url_decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8")).decode("utf-8")


def validate_password_strength(password: str):
    if not isinstance(password, str) or len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least 1 uppercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must include at least 1 number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return False, "Password must include at least 1 symbol."
    return True, "Password is valid."


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("Password cannot be empty.")

    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("utf-8")
    dk_b64 = base64.b64encode(dk).decode("utf-8")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${dk_b64}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iter_str, salt_b64, hash_b64 = stored_hash.split("$")
        if scheme != "pbkdf2_sha256":
            return False

        iterations = int(iter_str)
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected = base64.b64decode(hash_b64.encode("utf-8"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def init_auth_schema():
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(120) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'user',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            must_change_password BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_app_users_username
        ON app_users (username);
        """
    )
    cur.execute(
        """
        ALTER TABLE app_users
        ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER NOT NULL DEFAULT 0;
        """
    )
    cur.execute(
        """
        ALTER TABLE app_users
        ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ NULL;
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_events (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NULL REFERENCES app_users(id) ON DELETE SET NULL,
            username VARCHAR(120) NOT NULL,
            event_type VARCHAR(40) NOT NULL,
            event_status VARCHAR(20) NOT NULL DEFAULT 'success',
            event_detail TEXT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_auth_events_created_at
        ON auth_events (created_at DESC);
        """
    )
    conn.commit()
    cur.close()
    conn.close()


def log_auth_event(
    username: str,
    event_type: str,
    event_status: str = "success",
    event_detail: str = None,
    user_id: int = None,
):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO auth_events (user_id, username, event_type, event_status, event_detail)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (user_id, (username or "").strip().lower(), event_type, event_status, event_detail),
    )
    conn.commit()
    cur.close()
    conn.close()


def create_user(username: str, password: str, role: str = "user", must_change_password: bool = True):
    username = (username or "").strip().lower()
    if not username:
        return False, "Username is required."
    if role not in ("admin", "user"):
        return False, "Role must be 'admin' or 'user'."
    valid_password, password_msg = validate_password_strength(password)
    if not valid_password:
        return False, password_msg

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM app_users WHERE username = %s", (username,))
    if cur.fetchone():
        cur.close()
        conn.close()
        return False, "Username already exists."

    cur.execute(
        """
        INSERT INTO app_users (username, password_hash, role, is_active, must_change_password, updated_at)
        VALUES (%s, %s, %s, TRUE, %s, %s)
        """,
        (username, hash_password(password), role, must_change_password, _utc_now()),
    )
    conn.commit()
    cur.close()
    conn.close()
    return True, "User created."


def ensure_bootstrap_admin():
    admin_username = (os.getenv("APP_ADMIN_USERNAME") or "").strip().lower()
    admin_password = (os.getenv("APP_ADMIN_PASSWORD") or "").strip()
    if not admin_username or not admin_password:
        return

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM app_users WHERE username = %s", (admin_username,))
    exists = cur.fetchone() is not None
    if not exists:
        cur.execute(
            """
            INSERT INTO app_users (username, password_hash, role, is_active, must_change_password, updated_at)
            VALUES (%s, %s, 'admin', TRUE, FALSE, %s)
            """,
            (admin_username, hash_password(admin_password), _utc_now()),
        )
        conn.commit()
    cur.close()
    conn.close()


def authenticate_user(username: str, password: str):
    username = (username or "").strip().lower()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, password_hash, role, is_active, must_change_password, failed_login_attempts, locked_until
        FROM app_users
        WHERE username = %s
        """,
        (username,),
    )
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        log_auth_event(username, "sign_in", "failed", "Unknown username.")
        return False, "Invalid username or password.", None

    user_id, db_username, pwd_hash, role, is_active, must_change, failed_attempts, locked_until = row
    now = _utc_now()

    if locked_until and locked_until > now:
        mins = max(1, int((locked_until - now).total_seconds() // 60) + 1)
        cur.close()
        conn.close()
        log_auth_event(db_username, "sign_in", "failed", f"Account locked. Try again in {mins} minute(s).", user_id=user_id)
        return False, f"Account temporarily locked. Try again in {mins} minute(s).", None

    if not is_active:
        cur.close()
        conn.close()
        log_auth_event(db_username, "sign_in", "failed", "Account disabled.", user_id=user_id)
        return False, "This account is disabled.", None

    if not verify_password(password or "", pwd_hash):
        new_failed_attempts = int(failed_attempts or 0) + 1
        new_locked_until = None
        detail = "Invalid password."
        if new_failed_attempts >= MAX_LOGIN_ATTEMPTS:
            new_locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
            new_failed_attempts = 0
            detail = f"Too many attempts. Locked for {LOCKOUT_MINUTES} minute(s)."

        cur.execute(
            """
            UPDATE app_users
            SET failed_login_attempts = %s, locked_until = %s, updated_at = %s
            WHERE id = %s
            """,
            (new_failed_attempts, new_locked_until, now, user_id),
        )
        conn.commit()
        cur.close()
        conn.close()
        log_auth_event(db_username, "sign_in", "failed", detail, user_id=user_id)
        return False, "Invalid username or password.", None

    cur.execute(
        """
        UPDATE app_users
        SET failed_login_attempts = 0, locked_until = NULL, updated_at = %s
        WHERE id = %s
        """,
        (now, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    log_auth_event(db_username, "sign_in", "success", "User logged in.", user_id=user_id)

    return True, "Login successful.", {
        "id": user_id,
        "username": db_username,
        "role": role,
        "must_change_password": bool(must_change),
    }


def create_session_token(user: dict, max_age_hours: int = SESSION_MAX_AGE_HOURS):
    """
    Create signed session token for browser persistence across refresh.
    """
    if not user:
        return ""
    exp_ts = int((_utc_now() + timedelta(hours=max_age_hours)).timestamp())
    payload = f"{int(user.get('id', 0))}|{str(user.get('username', '')).lower()}|{exp_ts}"
    sig = _sign_payload(payload)
    token = f"{payload}|{sig}"
    return _b64url_encode(token)


def verify_session_token(token: str):
    """
    Validate token signature/expiry and return current user from DB if active.
    """
    if not token:
        return None
    try:
        decoded = _b64url_decode(token)
        parts = decoded.split("|")
        if len(parts) != 4:
            return None
        user_id_str, username, exp_ts_str, sig = parts
        payload = f"{user_id_str}|{username}|{exp_ts_str}"
        if not hmac.compare_digest(_sign_payload(payload), sig):
            return None
        exp_ts = int(exp_ts_str)
        if int(_utc_now().timestamp()) > exp_ts:
            return None
        user_id = int(user_id_str)
    except Exception:
        return None

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, role, is_active, must_change_password
        FROM app_users
        WHERE id = %s
        """,
        (user_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None

    db_user_id, db_username, role, is_active, must_change = row
    if not is_active or db_username.lower() != username.lower():
        return None

    return {
        "id": db_user_id,
        "username": db_username,
        "role": role,
        "must_change_password": bool(must_change),
    }


def set_password(username: str, new_password: str, must_change_password: bool = False):
    username = (username or "").strip().lower()
    valid_password, password_msg = validate_password_strength(new_password)
    if not valid_password:
        return False, password_msg

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_users
        SET password_hash = %s, must_change_password = %s, updated_at = %s
        WHERE username = %s
        """,
        (hash_password(new_password), must_change_password, _utc_now(), username),
    )
    updated = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()

    if updated == 0:
        return False, "User not found."
    return True, "Password updated."


def reset_password_by_manager(target_username: str, new_temp_password: str):
    ok, msg = set_password(target_username, new_temp_password, must_change_password=True)
    if ok:
        log_auth_event(target_username, "password_reset", "success", "Password reset by manager/admin.")
    return ok, msg


def list_users():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, role, is_active, must_change_password, failed_login_attempts, locked_until, created_at
        FROM app_users
        ORDER BY username ASC
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def delete_user_by_admin(target_username: str, acting_username: str):
    target_username = (target_username or "").strip().lower()
    acting_username = (acting_username or "").strip().lower()

    if not target_username:
        return False, "Select a valid user."
    if target_username == acting_username:
        return False, "You cannot delete your own active account."

    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, role FROM app_users WHERE username = %s", (target_username,))
    target = cur.fetchone()
    if not target:
        cur.close()
        conn.close()
        return False, "User not found."

    target_user_id, target_role = target
    if target_role == "admin":
        cur.execute("SELECT COUNT(*) FROM app_users WHERE role = 'admin' AND is_active = TRUE")
        admin_count = cur.fetchone()[0]
        if admin_count <= 1:
            cur.close()
            conn.close()
            return False, "Cannot delete the last active admin."

    cur.execute("DELETE FROM app_users WHERE id = %s", (target_user_id,))
    conn.commit()
    cur.close()
    conn.close()
    log_auth_event(acting_username, "delete_user", "success", f"Deleted user '{target_username}'.")
    return True, f"User '{target_username}' deleted."


def unlock_user_by_admin(target_username: str, acting_username: str):
    target_username = (target_username or "").strip().lower()
    acting_username = (acting_username or "").strip().lower()

    if not target_username:
        return False, "Select a valid user."

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_users
        SET failed_login_attempts = 0, locked_until = NULL, updated_at = %s
        WHERE username = %s
        """,
        (_utc_now(), target_username),
    )
    updated = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()

    if updated == 0:
        return False, "User not found."
    log_auth_event(acting_username, "unlock_user", "success", f"Unlocked user '{target_username}'.")
    return True, f"User '{target_username}' unlocked."


def set_user_active_by_admin(target_username: str, is_active: bool, acting_username: str):
    target_username = (target_username or "").strip().lower()
    acting_username = (acting_username or "").strip().lower()

    if not target_username:
        return False, "Select a valid user."
    if target_username == acting_username and not is_active:
        return False, "You cannot deactivate your own active account."

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, role FROM app_users WHERE username = %s", (target_username,))
    target = cur.fetchone()
    if not target:
        cur.close()
        conn.close()
        return False, "User not found."

    target_user_id, target_role = target
    if target_role == "admin" and not is_active:
        cur.execute("SELECT COUNT(*) FROM app_users WHERE role = 'admin' AND is_active = TRUE")
        admin_count = cur.fetchone()[0]
        if admin_count <= 1:
            cur.close()
            conn.close()
            return False, "Cannot deactivate the last active admin."

    cur.execute(
        """
        UPDATE app_users
        SET is_active = %s, updated_at = %s
        WHERE id = %s
        """,
        (is_active, _utc_now(), target_user_id),
    )
    conn.commit()
    cur.close()
    conn.close()

    state = "activated" if is_active else "deactivated"
    log_auth_event(acting_username, "toggle_user_active", "success", f"{state.title()} user '{target_username}'.")
    return True, f"User '{target_username}' {state}."


def record_sign_out(username: str, user_id: int = None):
    log_auth_event(username, "sign_out", "success", "User logged out.", user_id=user_id)


def get_auth_events(limit: int = 100):
    safe_limit = max(1, min(int(limit), 500))
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT created_at, username, event_type, event_status, COALESCE(event_detail, '')
        FROM auth_events
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (safe_limit,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
