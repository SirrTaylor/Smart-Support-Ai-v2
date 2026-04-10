import streamlit as st
import pandas as pd
import os
import uuid
import io
import time
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey
import psycopg2
from psycopg2.extras import Json
from auth import (
    init_auth_schema,
    ensure_bootstrap_admin,
    authenticate_user,
    create_user,
    set_password,
    reset_password_by_manager,
    delete_user_by_admin,
    list_users,
    record_sign_out,
    get_auth_events,
    unlock_user_by_admin,
    set_user_active_by_admin,
    create_session_token,
    verify_session_token,
    connect_postgres,
)

# 1. Setup and Environment
load_dotenv()


def _merge_streamlit_secrets_into_environ():
    """Streamlit Community Cloud secrets → os.environ (same names as .env)."""
    try:
        for key, val in st.secrets.items():
            if isinstance(val, str) and val.strip():
                if not os.getenv(key):
                    os.environ[key] = val
            elif isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, str) and sub_val.strip():
                        composite = f"{key}_{sub_key}".upper()
                        if not os.getenv(composite):
                            os.environ[composite] = sub_val
    except Exception:
        pass


_merge_streamlit_secrets_into_environ()

st.set_page_config(
    page_title=" Smart Support AI | Analytics Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CREDENTIALS ---
AZURE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")

DATABASE_NAME = "SupportAnalytics"
CONTAINER_NAME = "reviews"


# --- DIALOG POPUPS ---
# These three dialogs cover every outcome the user will encounter:
# success, error, and no results. They block the UI until OK is clicked
# so users always know when the app has finished processing.

@st.dialog("✅ Success")
def show_success_dialog(message: str):
    """Green success popup."""
    st.success(message)
    if st.session_state.pop("refresh_on_success_ok", False):
        st.rerun()

@st.dialog("❌ Error")
def show_error_dialog(message: str):
    """Red error popup."""
    st.error(message)

@st.dialog("ℹ️ No Results")
def show_no_results_dialog():
    """Neutral popup when analyst finds no matching data."""
    st.info("The analyst couldn't find relevant data for that query. Try a broader keyword like the company name.")


# --- UTILITY FUNCTIONS ---

def get_azure_client():
    return TextAnalyticsClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

def get_gpt_client():
    return AzureOpenAI(api_key=AOAI_KEY, api_version="2024-02-15-preview", azure_endpoint=AOAI_ENDPOINT)

def get_cosmos_container():
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.create_database_if_not_exists(id=DATABASE_NAME)
    container = db.create_container_if_not_exists(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path="/Company")
        # offer_throughput removed — not supported on serverless accounts
    )
    return container

def setup_postgres_schema(cur):
    """
    Extracted schema creation into a separate helper so it only runs once
    per connection, not once per row.
    """
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_support_data (
            id SERIAL PRIMARY KEY,
            sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_content JSONB
        );
    """)
    # Recreate the view to allow controlled column changes/order updates.
    # PostgreSQL does not allow changing existing view column positions/names
    # via CREATE OR REPLACE VIEW when the projection order changes.
    cur.execute("DROP VIEW IF EXISTS enriched_analytics_view;")
    cur.execute("""
        CREATE VIEW enriched_analytics_view AS
        SELECT
            id as internal_sync_id,
            sync_timestamp,
            COALESCE(
                data_content->>'source_date_iso',
                data_content->>'source_date',
                data_content->>'date',
                data_content->>'Date'
            ) as original_date,
            data_content->>'id' as record_id,
            data_content->>'Company' as company,
            data_content->>'sentiment' as sentiment,
            data_content->>'urgency' as urgency,
            (data_content->>'Rating')::numeric as rating,
            data_content->>'Review' as review_text
        FROM raw_support_data;
    """)

def mirror_to_postgres(enriched_batch):
    try:
        conn = connect_postgres()
        cur = conn.cursor()
        setup_postgres_schema(cur)

        for item in enriched_batch:
            cur.execute(
                "INSERT INTO raw_support_data (data_content) VALUES (%s)",
                [Json(item)]
            )

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        return str(e)  # Return error string so caller can show the correct dialog

def generate_executive_summary(items):
    gpt_client = get_gpt_client()
    summary_context = ""
    for i in items[:30]:
        summary_context += f"- [{i.get('Company')}] {i.get('sentiment')}: {str(i.get('review', i.get('Review')))[:150]}\n"

    prompt = f""""You are a Business Intelligence specialist. Don't just list facts. Identify the top 3 trends, any urgent risks, and give one recommendation based on the data provided. provide a 'Daily Intelligence Briefing'.
    Format it exactly like this:
    ### 📡 Daily Intelligence Briefing
    **Top 3 Strategic Trends:**
    1. [Trend 1]
    2. [Trend 2]
    3. [Trend 3]
    
    **Executive Recommendation:** [One clear sentence]
    
    Data:\n{summary_context}"""

    response = gpt_client.chat.completions.create(
        model=AOAI_DEPLOYMENT,
        messages=[{"role": "system", "content": "You provide concise executive intelligence."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def sanitize_company_name(name: str) -> str:
    """
    Strip characters that are unsafe in filenames and Cosmos partition keys.
    Keeps alphanumerics, spaces, hyphens, and underscores only.
    """
    import re
    return re.sub(r"[^\w\s\-]", "", name).strip()

def log_runtime_issue(scope: str, error: Exception):
    """Store recent runtime issues for quick in-app diagnostics."""
    if "runtime_issues" not in st.session_state:
        st.session_state.runtime_issues = []
    st.session_state.runtime_issues.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scope": scope,
        "error": str(error),
    })
    st.session_state.runtime_issues = st.session_state.runtime_issues[-20:]

def show_processing_overlay(placeholder, message="Processing..."):
    """Render a temporary full-screen processing overlay."""
    placeholder.markdown(
        f"""
        <style>
            .ssa-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(15, 23, 42, 0.28);
                z-index: 99999;
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 1;
                animation: ssa-fade-in 180ms ease-out;
            }}
            .ssa-card {{
                width: min(520px, 88vw);
                background: #ffffff;
                border-radius: 14px;
                padding: 20px 22px;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
                font-family: "Segoe UI", Arial, sans-serif;
                animation: ssa-card-in 220ms ease-out;
            }}
            .ssa-title {{
                margin: 0 0 10px 0;
                font-size: 18px;
                font-weight: 700;
                color: #0f172a;
            }}
            .ssa-sub {{
                margin: 0 0 12px 0;
                font-size: 13px;
                color: #475569;
            }}
            .ssa-track {{
                width: 100%;
                height: 10px;
                border-radius: 999px;
                background: #e2e8f0;
                overflow: hidden;
            }}
            .ssa-bar {{
                height: 100%;
                width: 40%;
                background: linear-gradient(90deg, #2563eb, #38bdf8);
                border-radius: 999px;
                animation: ssa-slide 1.2s infinite ease-in-out;
            }}
            @keyframes ssa-slide {{
                0% {{ transform: translateX(-120%); }}
                100% {{ transform: translateX(280%); }}
            }}
            @keyframes ssa-fade-in {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            @keyframes ssa-card-in {{
                from {{ opacity: 0; transform: translateY(6px) scale(0.985); }}
                to {{ opacity: 1; transform: translateY(0) scale(1); }}
            }}
        </style>
        <div class="ssa-overlay">
            <div class="ssa-card">
                <p class="ssa-title">Processing</p>
                <p class="ssa-sub">{message}</p>
                <div class="ssa-track"><div class="ssa-bar"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def hide_processing_overlay(placeholder):
    """Fade out overlay before clearing to avoid abrupt transitions."""
    placeholder.markdown(
        """
        <style>
            .ssa-overlay {
                animation: ssa-fade-out 220ms ease-in forwards !important;
            }
            .ssa-card {
                animation: ssa-card-out 220ms ease-in forwards !important;
            }
            @keyframes ssa-fade-out {
                from { opacity: 1; }
                to { opacity: 0; }
            }
            @keyframes ssa-card-out {
                from { opacity: 1; transform: translateY(0) scale(1); }
                to { opacity: 0; transform: translateY(6px) scale(0.985); }
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    time.sleep(0.22)
    placeholder.empty()

def master_sync_and_save(df, company_name, selected_date_col=None):
    """Unified function to Enrich, Save to Cosmos, and Mirror to Postgres."""
    container = get_cosmos_container()
    azure_client = get_azure_client()

    df.columns = [c.strip() for c in df.columns]
    rev_col = next((c for c in df.columns if any(k in c.lower() for k in ['review', 'review_text'])), None)

    if not rev_col:
        show_error_dialog("No review column found in the CSV. Please check your file and try again.")
        return False

    id_col = next((c for c in df.columns if 'id' in c.lower()), None)
    date_col = None
    if selected_date_col and selected_date_col in df.columns:
        date_col = selected_date_col
    else:
        date_col = next(
            (
                c for c in df.columns
                if any(k in c.lower() for k in ["date", "created", "time", "timestamp"])
                and "sync" not in c.lower()
            ),
            None
        )
    df = df.dropna(subset=[rev_col])

    enriched_batch = []
    st.info(f"Processing {len(df)} records for {company_name}...")
    progress_bar = st.progress(0)

    # Batch sentiment analysis — up to 10 docs per API call
    texts = [str(row[rev_col])[:5000] for _, row in df.iterrows()]
    sentiments = []
    batch_size = 10
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        try:
            responses = azure_client.analyze_sentiment(documents=batch)
            for resp in responses:
                sentiments.append(resp.sentiment.capitalize() if not resp.is_error else "Neutral")
        except Exception:
            sentiments.extend(["Neutral"] * len(batch))

    for i, (_, row) in enumerate(df.iterrows()):
        sentiment = sentiments[i] if i < len(sentiments) else "Neutral"

        rating_col = next((c for c in row.keys() if 'rating' in c.lower()), None)
        rating_val = pd.to_numeric(row.get(rating_col, 0), errors='coerce')

        urgency_flag = "Normal"
        if sentiment == "Negative":
            urgency_flag = "CRITICAL: ACTION REQUIRED" if rating_val <= 1 else "Attention Needed"

        doc = row.to_dict()
        raw_id = str(row[id_col]).strip() if id_col and pd.notnull(row[id_col]) else str(i)

        doc['id'] = f"{company_name.upper()}_{raw_id}"
        doc['Company'] = company_name.upper()
        doc['sentiment'] = sentiment
        doc['urgency'] = urgency_flag
        if date_col and date_col in row and pd.notnull(row[date_col]):
            raw_source_date = str(row[date_col]).strip()
            doc["source_date"] = raw_source_date
            parsed_source_date = pd.to_datetime(raw_source_date, errors="coerce")
            if pd.notna(parsed_source_date):
                doc["source_date_iso"] = parsed_source_date.strftime("%Y-%m-%d")

        for col, val in doc.items():
            if isinstance(val, (pd.Timestamp, datetime)):
                doc[col] = val.isoformat()

        enriched_batch.append(doc)
        progress_bar.progress((i + 1) / len(df))

    # Cloud Sync
    with st.spinner("Syncing to Azure Cosmos DB..."):
        for item in enriched_batch:
            container.upsert_item(item)

    # Postgres Mirror
    with st.spinner("Mirroring to PostgreSQL..."):
        pg_result = mirror_to_postgres(enriched_batch)

    # mirror_to_postgres returns True on success, error string on failure
    if pg_result is not True:
        show_error_dialog(f"PostgreSQL sync failed: {pg_result}")
        return False

    st.session_state.current_batch = enriched_batch
    return True


def render_auth_gate():
    """
    Blocks the app until a valid user logs in.
    Includes admin tools for creating users and resetting passwords.
    """
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
    if "auth_ready" not in st.session_state:
        st.session_state.auth_ready = False
    if "auth_token_checked" not in st.session_state:
        st.session_state.auth_token_checked = False

    if not st.session_state.auth_ready:
        try:
            init_auth_schema()
            ensure_bootstrap_admin()
            st.session_state.auth_ready = True
        except Exception as e:
            st.error("**Authentication setup failed:** could not connect to PostgreSQL.")
            st.code(str(e), language="text")
            pg_host = (os.getenv("POSTGRES_HOST") or "").strip().lower()
            if pg_host in ("", "localhost", "127.0.0.1", "::1"):
                st.markdown(
                    """
**Hosting on Streamlit Community Cloud?**  
There is no Postgres on `localhost` in that environment. Use a **hosted** database and put the same values in **App settings → Secrets** (or your `.env` locally):

| Secret | Example |
|--------|---------|
| `POSTGRES_HOST` | e.g. `ep-cool-name-12345.us-east-2.aws.neon.tech` (Neon) or your Azure/Supabase host |
| `POSTGRES_PORT` | usually `5432` |
| `POSTGRES_DB` | database name |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` | from the provider |

For **Neon**, **Supabase**, or **Azure Database for PostgreSQL**, add:

`POSTGRES_SSLMODE` = `require`

Then **Redeploy** the app. Allow **SSL** on the database; if your provider uses IP allowlists, allow **Streamlit Cloud egress** (or use “allow all” for a small demo DB only).
                    """
                )
            else:
                st.caption("Check host, port, password, firewall, and SSL settings (`POSTGRES_SSLMODE`).")
            st.stop()

    # Restore user from signed session token (survives browser refresh)
    if st.session_state.auth_user is None and not st.session_state.auth_token_checked:
        st.session_state.auth_token_checked = True
        token = st.query_params.get("session")
        restored_user = verify_session_token(token) if token else None
        if restored_user:
            st.session_state.auth_user = restored_user

    user = st.session_state.auth_user

    # Not logged in: show login-only screen.
    if not user:
        st.markdown("<h1 style='text-align: center;'>🔐 Smart Support AI Login</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Authorized company users only</p>", unsafe_allow_html=True)
        st.divider()

        with st.container(border=True):
            with st.form("login_form"):
                username = st.text_input("Username").strip()
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", width="stretch")

            if submitted:
                ok, message, auth_user = authenticate_user(username, password)
                if ok:
                    st.session_state.auth_user = auth_user
                    st.query_params["session"] = create_session_token(auth_user)
                    st.rerun()
                else:
                    st.error(message)

        st.info("If you forgot your password, contact your manager/admin for a reset.")
        st.stop()

    # Logged in sidebar
    with st.sidebar:
        st.markdown(f"**Signed in as:** `{user['username']}`")
        st.caption(f"Role: {user['role']}")
        if st.button("Logout", width="stretch"):
            record_sign_out(user["username"], user_id=user["id"])
            st.session_state.auth_user = None
            st.query_params.pop("session", None)
            st.rerun()

    # Force first-time password change when required.
    if user.get("must_change_password"):
        st.warning("You must change your temporary password before using the app.")
        with st.container(border=True):
            st.caption("Password rules: at least 8 characters, 1 uppercase letter, 1 number, and 1 symbol.")
            with st.form("force_change_password"):
                new_pw = st.text_input("New Password", type="password")
                confirm_pw = st.text_input("Confirm New Password", type="password")
                change_submitted = st.form_submit_button("Update Password", width="stretch")

            if change_submitted:
                if new_pw != confirm_pw:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = set_password(user["username"], new_pw, must_change_password=False)
                    if ok:
                        st.success("Password updated. Please log in again.")
                        st.session_state.auth_user = None
                        st.query_params.pop("session", None)
                        st.rerun()
                    else:
                        st.error(msg)
        st.stop()

    # Optional admin-only user management.
    if user.get("role") == "admin":
        with st.sidebar.expander("User Management", expanded=False):
            st.caption("Create, reset, and remove users. Password rules: 8+ chars, 1 uppercase, 1 number, 1 symbol.")

            with st.form("admin_create_user"):
                new_username = st.text_input("New Username").strip().lower()
                new_password = st.text_input("Temporary Password", type="password")
                new_role = st.selectbox("Role", options=["user", "admin"], index=0)
                create_submitted = st.form_submit_button("Create User", width="stretch")
            if create_submitted:
                ok, msg = create_user(
                    username=new_username,
                    password=new_password,
                    role=new_role,
                    must_change_password=True,
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

            with st.form("admin_reset_user_password"):
                target_username = st.text_input("Username to Reset").strip().lower()
                temp_password = st.text_input("New Temp Password", type="password")
                reset_submitted = st.form_submit_button("Reset Password", width="stretch")
            if reset_submitted:
                ok, msg = reset_password_by_manager(target_username, temp_password)
                if ok:
                    st.success("Password reset. User must change password at next login.")
                else:
                    st.error(msg)

            users_data = list_users()
            all_usernames = [u[0] for u in users_data]

            with st.form("admin_unlock_user"):
                unlock_target = st.selectbox(
                    "Unlock User",
                    options=all_usernames if all_usernames else ["No users found"],
                )
                unlock_submitted = st.form_submit_button("Unlock Account", width="stretch")
            if unlock_submitted:
                if unlock_target == "No users found":
                    st.error("No users available to unlock.")
                else:
                    ok, msg = unlock_user_by_admin(unlock_target, user["username"])
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            with st.form("admin_toggle_active"):
                toggle_target = st.selectbox(
                    "Enable/Disable User",
                    options=all_usernames if all_usernames else ["No users found"],
                )
                toggle_action = st.selectbox("Action", options=["Deactivate", "Activate"], index=0)
                toggle_submitted = st.form_submit_button("Apply Status", width="stretch")
            if toggle_submitted:
                if toggle_target == "No users found":
                    st.error("No users available.")
                else:
                    desired_active = toggle_action == "Activate"
                    ok, msg = set_user_active_by_admin(toggle_target, desired_active, user["username"])
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            with st.form("admin_delete_user"):
                deletable = [u for u in all_usernames if u != user["username"]]
                delete_target = st.selectbox(
                    "Delete User",
                    options=deletable if deletable else ["No deletable users"],
                )
                delete_submitted = st.form_submit_button("Remove User", width="stretch")

            if delete_submitted:
                if delete_target == "No deletable users":
                    st.error("No users available to delete.")
                else:
                    ok, msg = delete_user_by_admin(delete_target, user["username"])
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

            st.markdown("**Recent Auth Activity**")
            events = get_auth_events(limit=50)
            if events:
                event_df = pd.DataFrame(
                    events,
                    columns=["timestamp", "username", "event", "status", "detail"],
                )
                st.dataframe(event_df, width="stretch", hide_index=True, height=260)
                auth_csv = event_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download Auth Logs (CSV)",
                    data=auth_csv,
                    file_name=f"auth_events_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    width="stretch",
                )
            else:
                st.caption("No auth events recorded yet.")


# --- UI START ---
render_auth_gate()

st.markdown("<h1 style='text-align: center;'>💎 Smart Support AI: Analytics Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Analytics with Multi-Cloud Sync</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'><b>Environment:</b> Azure Cosmos | PostgreSQL | GPT-4 Turbo</p>", unsafe_allow_html=True)
st.divider()

# --- 1. INTELLIGENCE BRIEFING (Full Width) ---
with st.container(border=True):
    try:
        if 'exec_summary' in st.session_state:
            st.markdown(st.session_state.exec_summary)
        else:
            container = get_cosmos_container()
            all_items = list(container.query_items(
                query="""
                    SELECT TOP 200
                        c.Company,
                        c.sentiment,
                        c.Review,
                        c.review
                    FROM c
                    ORDER BY c._ts DESC
                """,
                enable_cross_partition_query=True
            ))
            if all_items:
                with st.spinner("Analyzing cross-company trends..."):
                    st.session_state.exec_summary = generate_executive_summary(all_items)
                st.markdown(st.session_state.exec_summary)
            else:
                st.info("👋 Welcome! Upload a CSV to start the sync and generate insights.")
    except Exception as e:
        log_runtime_issue("intelligence_briefing", e)
        st.caption("Intelligence Briefing unavailable.")

# --- 2. MASTER SYNC ---
st.subheader("⬆️ Master Sync")
with st.container(border=True):
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        sep_map = {",": ",", ";": ";", "Tab": "\t", "Auto-detect": None}
        sync_col1, sync_col2 = st.columns(2)
        with sync_col1:
            sep_choice = st.selectbox("Separator", [",", ";", "Tab", "Auto-detect"])
        with sync_col2:
            raw_comp_name = st.text_input("Company Name", placeholder="e.g., Netflix").strip()
            comp_name = sanitize_company_name(raw_comp_name)

        selected_date_col = None
        preview_df = None
        current_sep = sep_map[sep_choice]
        try:
            uploaded_file.seek(0)
            preview_df = pd.read_csv(
                uploaded_file,
                sep=current_sep if current_sep else None,
                engine='python' if not current_sep else None,
                nrows=80
            )
            uploaded_file.seek(0)
        except Exception as e:
            log_runtime_issue("csv_preview_read", e)
            st.warning("Could not preview CSV columns. You can still continue with auto-detect.")

        if preview_df is not None and not preview_df.empty:
            candidate_cols = [
                c for c in preview_df.columns
                if any(k in c.lower() for k in ["date", "created", "time", "timestamp"])
                and "sync" not in c.lower()
            ]
            date_options = ["Auto-detect"] + preview_df.columns.tolist()
            default_index = 0
            if candidate_cols:
                default_index = date_options.index(candidate_cols[0])
                st.caption(f"Detected date-like columns: {', '.join(candidate_cols)}")
            selected_option = st.selectbox(
                "Source Date Column (recommended for trend accuracy)",
                options=date_options,
                index=default_index
            )
            selected_date_col = None if selected_option == "Auto-detect" else selected_option

        st.divider()
        is_confirmed = st.checkbox("Confirm details and sync target")

        if is_confirmed and comp_name:
            if st.button("🚀 Sync Data and Save", width="stretch"):
                overlay_placeholder = st.empty()
                show_processing_overlay(overlay_placeholder, "Syncing data to Cosmos and PostgreSQL...")
                try:
                    with st.spinner("📂 Reading CSV file..."):
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file,
                            sep=current_sep if current_sep else None,
                            engine='python' if not current_sep else None
                        )

                    result = master_sync_and_save(df, comp_name, selected_date_col=selected_date_col)

                    if result:
                        if 'exec_summary' in st.session_state:
                            del st.session_state.exec_summary
                        st.session_state.last_sync_company = comp_name
                        # SUCCESS POPUP — shown after sync completes
                        show_success_dialog(
                            f"Sync complete: {len(df)} records for **{comp_name}** were saved successfully."
                        )

                except Exception as e:
                    log_runtime_issue("master_sync", e)
                    show_error_dialog(f"Sync failed: {e}")
                finally:
                    hide_processing_overlay(overlay_placeholder)

        elif not comp_name and is_confirmed:
            st.warning("Please enter a Company Name.")

st.divider()

# --- 3. AI COMMAND CENTER ---
from ai_command_center import (
    query_cosmos_analysis,
    run_analyst,
    build_show_only_response,
    parse_user_intent,
    compute_root_cause_clusters,
    detect_anomalies,
)

st.subheader("💬 AI Command Center")
with st.container(border=True):
    if "ai_memory_context" not in st.session_state:
        st.session_state.ai_memory_context = None
    if "ai_last_results_df" not in st.session_state:
        st.session_state.ai_last_results_df = None

    query = st.text_input("Ask about your data...", placeholder="e.g., What are the main complaints for Netflix?")
    st.caption("Tip: Use 'show' for direct results, or ask for 'trend', 'anomaly', 'root cause', or 'recommended actions'.")
    ai_col1, ai_col2 = st.columns([3, 1])
    with ai_col1:
        use_followup_memory = st.checkbox("Use follow-up memory", value=True)
    with ai_col2:
        if st.button("Clear AI Memory", width="stretch"):
            st.session_state.ai_memory_context = None
            st.session_state.ai_last_results_df = None
            st.success("AI memory cleared.")

    if st.button("Execute Analysis", width="stretch") and query:
        overlay_placeholder = st.empty()
        show_processing_overlay(overlay_placeholder, "Analyzing records and preparing AI briefing...")
        intent = parse_user_intent(query)
        memory_ctx = st.session_state.ai_memory_context if use_followup_memory else None

        # Step 1: Fetch matching records from Cosmos DB
        try:
            with st.spinner("🔍 Fetching relevant records from Cosmos DB..."):
                try:
                    results_df, new_context = query_cosmos_analysis(query, memory_context=memory_ctx)
                except RuntimeError as e:
                    # DB connection or query failure — show error popup and stop
                    show_error_dialog(str(e))
                    st.stop()

            # Step 2: Run GPT analyst on the results
            if results_df is not None and not results_df.empty:
                st.session_state.ai_memory_context = new_context
                st.session_state.ai_last_results_df = results_df.copy()

                if intent.get("show_only") and not any([
                    intent.get("wants_actions"),
                    intent.get("wants_trend_chart"),
                    intent.get("wants_anomaly"),
                    intent.get("wants_root_cause"),
                ]):
                    analyst_report = build_show_only_response(query, results_df)
                else:
                    with st.spinner("🧠 Senior Analyst is reviewing records..."):
                        analyst_report = run_analyst(
                            query,
                            results_df,
                            memory_context=st.session_state.ai_memory_context,
                        )

                st.markdown("### 📊 Analyst Briefing")
                with st.container(border=True):
                    st.markdown(analyst_report)

                if intent.get("wants_root_cause"):
                    clusters = compute_root_cause_clusters(results_df, top_n=6)
                    if clusters:
                        st.markdown("### 🧩 Root Cause Themes")
                        cluster_df = pd.DataFrame(clusters, columns=["theme", "mentions"])
                        st.dataframe(cluster_df, width="stretch", hide_index=True)

                if intent.get("wants_anomaly"):
                    anomaly_msg = detect_anomalies(results_df)
                    if anomaly_msg:
                        st.warning(anomaly_msg)
                    else:
                        st.info("No clear anomaly detected in the current result set.")

                if intent.get("wants_trend_chart"):
                    trend_df = results_df.copy()
                    source_series = None
                    if "source_date" in trend_df.columns:
                        source_series = pd.to_datetime(trend_df["source_date"], errors="coerce")

                    if source_series is not None and source_series.notna().any():
                        trend_df["trend_date"] = source_series
                    else:
                        trend_df["trend_date"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")

                    trend_df = trend_df.dropna(subset=["trend_date"])
                    if not trend_df.empty:
                        trend_df["is_negative"] = (trend_df["sentiment"].astype(str).str.lower() == "negative").astype(int)
                        daily = trend_df.groupby(trend_df["trend_date"].dt.date).agg(
                            avg_rating=("rating", "mean"),
                            negative_reviews=("is_negative", "sum"),
                            total_reviews=("is_negative", "count"),
                        ).reset_index()
                        daily["negative_rate"] = daily["negative_reviews"] / daily["total_reviews"]
                        daily = daily.rename(columns={"trend_date": "Date"})
                        daily = daily.sort_values("Date")
                        st.markdown("### 📈 Trend Chart")
                        st.line_chart(
                            daily.set_index("Date")[["avg_rating", "negative_rate"]],
                            width="stretch",
                        )

                export_csv = results_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Export This Answer Data (CSV)",
                    data=export_csv,
                    file_name=f"ai_answer_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    width="stretch",
                )

                with st.expander(f"📂 Review Source Data ({len(results_df)} records used)"):
                    st.dataframe(results_df, width="stretch", hide_index=True, height=300)
            else:
                # NO RESULTS POPUP — query ran fine but nothing matched
                show_no_results_dialog()
        finally:
            hide_processing_overlay(overlay_placeholder)

# --- GLOBAL PERSISTENT FILTER & DOWNLOAD ---
st.divider()
st.subheader("🔍 Filter and Download")

with st.container(border=True):
    all_data_df = pd.DataFrame()
    data_source = None

    try:
        with connect_postgres() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM enriched_analytics_view")
                pg_rows = cur.fetchall()
                pg_cols = [desc[0] for desc in (cur.description or [])]
                all_data_df = pd.DataFrame(pg_rows, columns=pg_cols)

        all_data_df.columns = [c.strip() for c in all_data_df.columns]
        data_source = "postgres"
    except psycopg2.Error as e:
        log_runtime_issue("postgres_filter_panel", e)
        # Keep the panel usable when PG is temporarily unavailable.
        if 'current_batch' in st.session_state:
            all_data_df = pd.DataFrame(st.session_state.current_batch)
            all_data_df.columns = [c.strip().lower() for c in all_data_df.columns]
            data_source = "session"
    except Exception as e:
        log_runtime_issue("filter_panel_unexpected", e)
        # Keep the panel usable when PG is temporarily unavailable.
        if 'current_batch' in st.session_state:
            all_data_df = pd.DataFrame(st.session_state.current_batch)
            all_data_df.columns = [c.strip().lower() for c in all_data_df.columns]
            data_source = "session"

    if all_data_df.empty:
        st.info("No enriched data found yet. Sync a CSV first to populate this section.")
    else:
        if data_source == "session":
            st.caption("Showing current session data (PostgreSQL view unavailable).")

        f_col1, f_col2 = st.columns(2)
        f_col3, f_col4 = st.columns(2)

        # Normalize expected columns if needed.
        if "Company" in all_data_df.columns and "company" not in all_data_df.columns:
            all_data_df["company"] = all_data_df["Company"]
        if "sentiment" not in all_data_df.columns:
            all_data_df["sentiment"] = ""
        if "Rating" in all_data_df.columns and "rating" not in all_data_df.columns:
            all_data_df["rating"] = all_data_df["Rating"]
        if "timestamp" in all_data_df.columns and "sync_timestamp" not in all_data_df.columns:
            all_data_df["sync_timestamp"] = all_data_df["timestamp"]

        # 1. Company Filter
        companies = sorted(all_data_df['company'].dropna().unique().tolist())
        selected_companies = f_col1.multiselect("Filter by Company", options=companies, placeholder="Choose companies...")

        # 2. Sentiment Filter
        sentiments = sorted(all_data_df['sentiment'].dropna().unique().tolist())
        selected_sentiments = f_col2.multiselect("Filter by Sentiment", options=sentiments, placeholder="Choose sentiments...")

        # 3. Rating Filter
        all_data_df['rating'] = pd.to_numeric(all_data_df['rating'], errors='coerce').fillna(0).astype(int)
        ratings = sorted(all_data_df['rating'].unique().tolist())
        selected_ratings = f_col3.multiselect("Filter by Rating", options=ratings, placeholder="Choose ratings...")

        # 4. Date Filter (source/original date only; never use sync timestamp)
        date_range = None
        apply_date_filter = False
        date_filter_col = None
        if "original_date" in all_data_df.columns:
            all_data_df["filter_date"] = pd.to_datetime(all_data_df["original_date"], errors="coerce").dt.date
            if all_data_df["filter_date"].notna().any():
                date_filter_col = "filter_date"

        if date_filter_col:
            valid_dates = all_data_df[date_filter_col].dropna()
            if not valid_dates.empty:
                min_d = valid_dates.min()
                max_d = valid_dates.max()
                apply_date_filter = f_col4.checkbox("Apply Date Filter", value=False)
                date_range = f_col4.date_input(
                    "Filter by Date Range",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d
                )
            else:
                f_col4.info("No valid source dates found.")
        else:
            f_col4.info("No source date field found.")

        # Apply Filters
        filtered_df = all_data_df.copy()
        if selected_companies:
            filtered_df = filtered_df[filtered_df['company'].isin(selected_companies)]
        if selected_sentiments:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(selected_sentiments)]
        if selected_ratings:
            filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]

        if apply_date_filter and isinstance(date_range, tuple) and len(date_range) == 2 and "filter_date" in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df["filter_date"] >= date_range[0]) &
                (filtered_df["filter_date"] <= date_range[1])
            ]

        st.dataframe(filtered_df, width="stretch", hide_index=True, height=400)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            label=f"📥 Download Filtered Export ({len(filtered_df)} records)",
            data=csv_data,
            file_name=f"Global_Export_{timestamp_str}.csv",
            mime="text/csv",
            width="stretch"
        )

# --- DANGER ZONE ---
with st.expander("⚠️ Danger Zone"):
    st.warning("These actions are irreversible. Proceed with caution.")

    delete_col1, delete_col2 = st.columns(2)

    # --- SCOPED DELETE (by Company) ---
    with delete_col1:
        st.subheader("Delete by Company")
        try:
            container = get_cosmos_container()
            available_companies = sorted(list(container.query_items(
                query="""
                    SELECT DISTINCT VALUE c.Company
                    FROM c
                    WHERE IS_DEFINED(c.Company) AND c.Company != ""
                """,
                enable_cross_partition_query=True
            )))
        except Exception as e:
            log_runtime_issue("delete_companies_list", e)
            available_companies = []

        company_to_delete = st.selectbox("Select Company to Delete", options=available_companies or ["No data found"])
        confirm_scoped = st.checkbox("I confirm I want to delete this company's data", key="confirm_scoped")

        if confirm_scoped and st.button("🗑️ Delete Company Data", width="stretch", key="btn_scoped"):
            if company_to_delete and company_to_delete != "No data found":
                with st.spinner(f"Deleting all data for {company_to_delete}..."):
                    try:
                        # Delete from Cosmos
                        container = get_cosmos_container()
                        items_to_delete = list(container.query_items(
                            query="""
                                SELECT c.id, c.Company
                                FROM c
                                WHERE c.Company = @company
                            """,
                            parameters=[{"name": "@company", "value": company_to_delete}],
                            enable_cross_partition_query=True
                        ))
                        for item in items_to_delete:
                            container.delete_item(item=item['id'], partition_key=item['Company'])

                        # Delete from Postgres
                        conn = connect_postgres()
                        cur = conn.cursor()
                        cur.execute("""
                            DELETE FROM raw_support_data
                            WHERE data_content->>'Company' = %s
                        """, (company_to_delete,))
                        conn.commit()
                        cur.close()
                        conn.close()

                        # Clear session state
                        if 'current_batch' in st.session_state:
                            del st.session_state.current_batch
                        if 'exec_summary' in st.session_state:
                            del st.session_state.exec_summary

                        # SUCCESS POPUP + refresh so decoupled sections reload from DB state
                        st.session_state.refresh_on_success_ok = True
                        show_success_dialog(f"All data for **{company_to_delete}** has been deleted successfully.")

                    except Exception as e:
                        log_runtime_issue("scoped_delete", e)
                        show_error_dialog(f"Delete failed: {e}")

    # --- NUCLEAR DELETE (Full Wipe) ---
    with delete_col2:
        st.subheader("Delete All Data")
        confirm_text = st.text_input("Type DELETE to confirm full wipe", key="confirm_nuke")
        confirm_nuke = st.checkbox("I understand this cannot be undone", key="confirm_nuke_check")

        if confirm_nuke and confirm_text == "DELETE":
            if st.button("💣 Wipe All Data", width="stretch", key="btn_nuke"):
                with st.spinner("Wiping all data from Cosmos DB and PostgreSQL..."):
                    try:
                        # Wipe Cosmos — delete and recreate the container
                        cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
                        db = cosmos_client.get_database_client(DATABASE_NAME)
                        db.delete_container(CONTAINER_NAME)
                        db.create_container(
                            id=CONTAINER_NAME,
                            partition_key=PartitionKey(path="/Company")
                        )

                        # Wipe Postgres
                        conn = connect_postgres()
                        cur = conn.cursor()
                        cur.execute("TRUNCATE TABLE raw_support_data RESTART IDENTITY;")
                        conn.commit()
                        cur.close()
                        conn.close()

                        # Clear all session state
                        for key in ['current_batch', 'exec_summary', 'last_sync_company']:
                            st.session_state.pop(key, None)

                        # SUCCESS POPUP + refresh so decoupled sections reload from DB state
                        st.session_state.refresh_on_success_ok = True
                        show_success_dialog("All data has been wiped successfully.")

                    except Exception as e:
                        log_runtime_issue("full_wipe", e)
                        show_error_dialog(f"Full wipe failed: {e}")

if st.session_state.get("runtime_issues"):
    with st.sidebar.expander("Runtime Diagnostics", expanded=False):
        diag_df = pd.DataFrame(st.session_state.get("runtime_issues", []))
        if not diag_df.empty:
            st.dataframe(diag_df.tail(10), width="stretch", hide_index=True, height=240)

st.divider()
st.markdown("<center><p style='color: gray;'>Built by Brendy Pro | 2026 | Enterprise Multi-Cloud Edition</p></center>", unsafe_allow_html=True)