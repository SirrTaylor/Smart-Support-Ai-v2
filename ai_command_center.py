# ai_command_center.py
import os
import re
import json
import pandas as pd
from collections import Counter
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey


# --- CLIENTS ---

AZURE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")

DATABASE_NAME = "SupportAnalytics"
CONTAINER_NAME = "reviews"



def get_gpt_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def get_cosmos_container():
    client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
    db = client.create_database_if_not_exists(id=DATABASE_NAME)
    container = db.create_container_if_not_exists(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path="/Company")
        # offer_throughput removed — not supported on serverless accounts
    )
    return container

# --- INPUT SANITIZER ---

def sanitize_query_input(text: str) -> str:
    """Strip characters that could cause Cosmos query or GPT parsing issues."""
    text = re.sub(r"[''`]", "", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = re.sub(r"[^\w\s,]", " ", text)
    return text.strip()


# --- KEYWORD EXTRACTOR ---

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "from",
    "show", "give", "tell", "about", "please", "can", "you", "me", "what", "which",
    "are", "is", "was", "were", "do", "does", "did", "all", "any", "latest", "recent"
}

DATE_NOISE_TERMS = {
    "trend", "trends", "timeline", "time", "over", "period", "between", "from", "to", "and",
    "month", "months", "year", "years", "daily", "weekly", "monthly", "quarter", "quarterly"
}

GENERIC_CONTENT_TERMS = {
    "review", "reviews", "complaint", "complaints", "feedback", "comment", "comments",
    "data", "records", "record", "insight", "insights", "analysis"
}

SENTIMENT_KEYWORDS = {
    "positive": "Positive",
    "good": "Positive",
    "great": "Positive",
    "happy": "Positive",
    "negative": "Negative",
    "bad": "Negative",
    "poor": "Negative",
    "angry": "Negative",
    "neutral": "Neutral",
    "mixed": "Neutral",
}

FOLLOW_UP_PREFIXES = (
    "and ",
    "also ",
    "what about ",
    "how about ",
    "now show ",
    "same for ",
)


def _extract_json_object(text: str):
    """
    Parse JSON robustly from model output, including fenced blocks or extra text.
    """
    if not text:
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _fallback_extract_terms(user_query: str):
    """
    Deterministic keyword fallback when GPT extraction fails.
    Keeps AI command center responsive instead of returning no-result immediately.
    """
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", user_query)]
    terms = []
    for w in words:
        # Ignore stopwords, date/time wording, and pure year tokens (e.g. 2025)
        if (
            len(w) < 3
            or w in STOPWORDS
            or w in DATE_NOISE_TERMS
            or re.fullmatch(r"20\d{2}", w)
        ):
            continue
        terms.append(w)
    # De-duplicate while preserving order
    terms = list(dict.fromkeys(terms))
    return [], terms[:6]


def _extract_requested_sentiment(user_query: str):
    query_lower = user_query.lower()
    for key, canonical in SENTIMENT_KEYWORDS.items():
        if key in query_lower:
            return canonical
    return None


def _is_follow_up_query(user_query: str):
    q = (user_query or "").strip().lower()
    return any(q.startswith(prefix) for prefix in FOLLOW_UP_PREFIXES)


def parse_user_intent(user_query: str):
    q = (user_query or "").strip().lower()
    show_only = (
        any(k in q for k in ["show", "list", "display", "give me"])
        and not any(k in q for k in ["recommend", "action", "what should", "next step", "why"])
    )
    wants_trend_chart = any(k in q for k in ["trend", "over time", "timeline", "chart", "graph"])
    wants_actions = any(k in q for k in ["recommend", "action", "what should", "next step"])
    wants_anomaly = any(k in q for k in ["anomaly", "spike", "unusual", "outlier", "sudden"])
    wants_root_cause = any(k in q for k in ["root cause", "why", "drivers", "theme", "cluster"])
    return {
        "show_only": show_only,
        "wants_trend_chart": wants_trend_chart,
        "wants_actions": wants_actions,
        "wants_anomaly": wants_anomaly,
        "wants_root_cause": wants_root_cause,
        "is_follow_up": _is_follow_up_query(q),
    }


def extract_requested_date_range(user_query: str):
    """
    Extract a requested date range from natural language query.
    Supports:
    - "between 2025 and 2026"
    - "from 2025 to 2026"
    - "2025-2026"
    - single year references (e.g., "in 2025")
    - explicit ISO dates (YYYY-MM-DD)
    Returns: (start_ts, end_ts) as pandas Timestamps or (None, None)
    """
    q = (user_query or "").lower().strip()
    if not q:
        return None, None

    # Explicit dates first
    iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if len(iso_dates) >= 2:
        start = pd.to_datetime(iso_dates[0], errors="coerce")
        end = pd.to_datetime(iso_dates[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            if end < start:
                start, end = end, start
            return start, end

    # Year range patterns
    year_range_match = re.search(
        r"(?:between|from)?\s*(20\d{2})\s*(?:and|to|-)\s*(20\d{2})",
        q
    )
    if year_range_match:
        y1 = int(year_range_match.group(1))
        y2 = int(year_range_match.group(2))
        start_year, end_year = sorted([y1, y2])
        start = pd.Timestamp(f"{start_year}-01-01")
        end = pd.Timestamp(f"{end_year}-12-31 23:59:59")
        return start, end

    # Single year fallback
    one_year = re.search(r"\b(20\d{2})\b", q)
    if one_year:
        y = int(one_year.group(1))
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y}-12-31 23:59:59")
        return start, end

    return None, None

def extract_keywords_from_query(user_query: str):
    """
    Uses GPT to extract structured keywords from a natural language query.
    Returns a tuple: (companies: list, terms: list)
    """
    gpt_client = get_gpt_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    try:
        response = gpt_client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract searchable keywords from the query. "
                        "Return a JSON object with two keys: "
                        "'companies' (list of company names, always uppercase) and "
                        "'terms' (list of other content search terms like "
                        "'complaints', 'positive', 'slow', 'billing'). "
                        "IGNORE: time words (trend, over time, month, year), "
                        "comparison words (vs, best, worst, compare), "
                        "vague words (everything, all, new, why, tell me), "
                        "financial words (revenue, NPS, churn, score). "
                        "If nothing valid found return: {\"companies\": [], \"terms\": []} "
                        "Return ONLY valid JSON, no extra text. "
                        "Example: 'Compare Netflix and Amazon complaints' -> "
                        "{\"companies\": [\"NETFLIX\", \"AMAZON\"], \"terms\": [\"complaints\"]}"
                    )
                },
                {"role": "user", "content": user_query}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        parsed = _extract_json_object(raw)
        if not isinstance(parsed, dict):
            return _fallback_extract_terms(user_query)

        companies = parsed.get("companies", [])
        terms = parsed.get("terms", [])

        # Safety normalization
        companies = [str(c).strip().upper() for c in companies if str(c).strip()]
        terms = [str(t).strip().lower() for t in terms if str(t).strip()]
        return companies, terms

    except Exception:
        return _fallback_extract_terms(user_query)


# --- COSMOS DB QUERY ---


def query_cosmos_analysis(user_query: str, memory_context=None):
    """
    Full pipeline: sanitize -> extract keywords -> query Cosmos DB.
    Returns a DataFrame or None if no results / invalid query.
    """
    # Step 1: Sanitize
    clean_query = sanitize_query_input(user_query)

    # Step 2: Extract keywords
    companies, terms = extract_keywords_from_query(clean_query)
    requested_sentiment = _extract_requested_sentiment(clean_query)
    requested_start_date, requested_end_date = extract_requested_date_range(clean_query)
    # Remove date-related noise terms and plain year tokens from search terms so
    # time-window prompts don't accidentally over-restrict text matching.
    terms = [
        t for t in terms
        if t
        and t.lower() not in DATE_NOISE_TERMS
        and t.lower() not in GENERIC_CONTENT_TERMS
        and t.lower() not in SENTIMENT_KEYWORDS
        and not re.fullmatch(r"20\d{2}", str(t).strip())
    ]
    intent = parse_user_intent(clean_query)

    # Follow-up memory: if user query is a continuation, carry forward prior filters.
    if memory_context and intent["is_follow_up"]:
        companies = companies or memory_context.get("companies", [])
        terms = terms or memory_context.get("terms", [])
        if not requested_sentiment:
            requested_sentiment = memory_context.get("requested_sentiment")

    try:
        container = get_cosmos_container()
        # Safe broad fetch. All filtering is done locally to avoid fragile
        # Cosmos SQL function/type issues on mixed legacy documents.
        is_time_series_request = intent.get("wants_trend_chart") or intent.get("wants_anomaly")
        fetch_limit = 1500 if is_time_series_request else 600
        items = list(container.query_items(
            query=f"""
                SELECT
                    c.Company,
                    c.sentiment,
                    c.urgency,
                    c.Rating,
                    c.Review,
                    c.source_date_iso,
                    c.source_date,
                    c.date,
                    c.Date,
                    c._ts
                FROM c
                OFFSET 0 LIMIT {int(fetch_limit)}
            """,
            enable_cross_partition_query=True
        ))

        context = {
            "companies": companies,
            "terms": terms,
            "requested_sentiment": requested_sentiment,
            "requested_start_date": str(requested_start_date) if requested_start_date is not None else None,
            "requested_end_date": str(requested_end_date) if requested_end_date is not None else None,
            "intent": intent,
        }

        if not items:
            return None, context

        # Step 5: Normalise into a DataFrame
        rows = []
        for item in items:
            resolved_source_date = (
                item.get("source_date_iso")
                or item.get("source_date")
                or item.get("date")
                or item.get("Date")
                or ""
            )
            rows.append({
                "company":    item.get("Company", ""),
                "sentiment":  item.get("sentiment", ""),
                "urgency":    item.get("urgency", ""),
                "rating":     item.get("Rating", ""),
                "review":     item.get("Review", ""),
                "source_date": resolved_source_date,
                # Cosmos stores _ts as a Unix timestamp — convert to readable date
                "timestamp":  pd.to_datetime(item.get("_ts", 0), unit="s").strftime("%Y-%m-%d %H:%M")
            })

        results_df = pd.DataFrame(rows)

        # Defensive local filtering to preserve intent even when Cosmos falls
        # back to broader retrieval on compatibility errors.
        if not results_df.empty and companies:
            company_upper = [c.upper() for c in companies]
            results_df = results_df[
                results_df["company"].fillna("").astype(str).str.upper().apply(
                    lambda v: any(c in v for c in company_upper)
                )
            ]

        pre_term_df = results_df.copy()
        if not results_df.empty and terms:
            mask = pd.Series(True, index=results_df.index)
            for term in terms:
                t = str(term).lower()
                term_mask = (
                    results_df["company"].fillna("").astype(str).str.lower().str.contains(t, na=False)
                    | results_df["sentiment"].fillna("").astype(str).str.lower().str.contains(t, na=False)
                    | results_df["review"].fillna("").astype(str).str.lower().str.contains(t, na=False)
                )
                mask = mask & term_mask
            results_df = results_df[mask]

        if not results_df.empty and requested_sentiment:
            results_df = results_df[
                results_df["sentiment"].fillna("").astype(str).str.lower() == requested_sentiment.lower()
            ]
        elif results_df.empty and not pre_term_df.empty and (companies or requested_sentiment):
            # If terms over-constrain the set, fall back to company/sentiment intent.
            # This keeps natural prompts like "show negative reviews from X" reliable.
            results_df = pre_term_df
            if requested_sentiment:
                results_df = results_df[
                    results_df["sentiment"].fillna("").astype(str).str.lower() == requested_sentiment.lower()
                ]

        # Apply requested date range after retrieval.
        # This avoids Cosmos SQL compatibility issues across mixed legacy schemas.
        if not results_df.empty and (requested_start_date is not None or requested_end_date is not None):
            analysis_series = None
            if "source_date" in results_df.columns:
                analysis_series = pd.to_datetime(results_df["source_date"], errors="coerce")

            if analysis_series is None or not analysis_series.notna().any():
                analysis_series = pd.to_datetime(results_df["timestamp"], errors="coerce")

            results_df["analysis_date"] = analysis_series
            results_df = results_df.dropna(subset=["analysis_date"])

            if requested_start_date is not None:
                results_df = results_df[results_df["analysis_date"] >= requested_start_date]
            if requested_end_date is not None:
                results_df = results_df[results_df["analysis_date"] <= requested_end_date]

            results_df = results_df.drop(columns=["analysis_date"], errors="ignore")

        return results_df, context

    except Exception as e:
        raise RuntimeError(f"Cosmos DB query failed: {e}")


# --- GPT ANALYST ---

def compute_root_cause_clusters(results_df: pd.DataFrame, top_n: int = 5):
    if results_df is None or results_df.empty or "review" not in results_df.columns:
        return []

    stop = {
        "the", "and", "for", "that", "this", "with", "have", "from", "your", "about", "are", "was",
        "were", "been", "very", "just", "when", "what", "would", "there", "they", "them", "than",
        "into", "only", "also", "their", "could", "should", "after", "before", "because"
    }
    words = []
    for txt in results_df["review"].fillna("").astype(str).tolist():
        words.extend([w.lower() for w in re.findall(r"[A-Za-z]{3,}", txt)])

    counts = Counter([w for w in words if w not in stop])
    return counts.most_common(top_n)


def detect_anomalies(results_df: pd.DataFrame):
    if results_df is None or results_df.empty or "timestamp" not in results_df.columns:
        return None

    df = results_df.copy()
    source_series = None
    if "source_date" in df.columns:
        source_series = pd.to_datetime(df["source_date"], errors="coerce")

    if source_series is not None and source_series.notna().any():
        df["trend_date"] = source_series.dt.date
    else:
        df["trend_date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date

    df = df.dropna(subset=["trend_date"])
    if df.empty:
        return None

    daily = df.groupby("trend_date").agg(
        total=("sentiment", "count"),
        negative=("sentiment", lambda s: (s.astype(str).str.lower() == "negative").sum()),
    ).reset_index()
    daily["negative_rate"] = daily["negative"] / daily["total"]
    if len(daily) < 4:
        return None

    baseline = daily.iloc[:-1]["negative_rate"]
    latest = float(daily.iloc[-1]["negative_rate"])
    mean = float(baseline.mean())
    std = float(baseline.std(ddof=0))
    threshold = mean + (2 * std if std > 0 else 0.15)

    if latest > threshold and latest > mean:
        return (
            f"Anomaly detected: latest negative rate is {latest:.1%}, "
            f"above baseline {mean:.1%}."
        )
    return None


def build_show_only_response(user_query: str, results_df: pd.DataFrame) -> str:
    """
    Deterministic non-GPT response for explicit show/list queries.
    """
    if results_df is None or results_df.empty:
        return "No matching records found."

    out = []
    out.append(f"Showing {len(results_df)} matching records for: `{user_query}`.")

    if "company" in results_df.columns:
        top_companies = results_df["company"].fillna("").astype(str).value_counts().head(3)
        if not top_companies.empty:
            comp_parts = [f"{k} ({v})" for k, v in top_companies.items()]
            out.append(f"Top companies in results: {', '.join(comp_parts)}.")

    if "sentiment" in results_df.columns:
        sent_counts = results_df["sentiment"].fillna("").astype(str).value_counts()
        if not sent_counts.empty:
            sent_parts = [f"{k}: {v}" for k, v in sent_counts.items()]
            out.append(f"Sentiment breakdown: {', '.join(sent_parts)}.")

    if "rating" in results_df.columns:
        rating_series = pd.to_numeric(results_df["rating"], errors="coerce").dropna()
        if not rating_series.empty:
            out.append(f"Average rating in selection: {rating_series.mean():.2f}.")

    return "\n\n".join([f"- {line}" for line in out])


def run_analyst(user_query: str, results_df: pd.DataFrame, memory_context=None) -> str:
    """
    Sends a capped, truncated data snapshot to GPT and returns the analyst report.
    """
    gpt_client = get_gpt_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    intent = parse_user_intent(user_query)

    # Cap rows and truncate review text to protect context window
    MAX_ROWS = 20
    display_df = results_df.head(MAX_ROWS).copy()
    display_df['review'] = display_df['review'].str[:200]
    data_snapshot = display_df[['company', 'sentiment', 'rating', 'review']].to_string(index=False)

    behavior_rules = (
        "If user intent is 'show/list/display only', respond with concise factual bullets and NO recommendations. "
        if intent["show_only"]
        else "Include recommendations only when user explicitly asks for actions/recommendations. "
    )

    if memory_context:
        memory_note = (
            f"Prior context filters: companies={memory_context.get('companies', [])}, "
            f"terms={memory_context.get('terms', [])}, "
            f"requested_sentiment={memory_context.get('requested_sentiment')}."
        )
    else:
        memory_note = "No prior context."

    response = gpt_client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Senior Data Analyst. Your available data schema contains ONLY: "
                    "company name, sentiment (Positive/Negative/Neutral), rating (1-5), "
                    "review text, urgency flag, and timestamp. "
                    "If the user asks about metrics NOT in this schema (revenue, churn, NPS, "
                    "product features, financial impact), explicitly state that this data is "
                    "not available in the current dataset and suggest what data would be needed. "
                    "Never invent or estimate figures. "
                    "For available data: break findings by company, identify sentiment distribution, "
                    "highlight keywords from reviews. "
                    f"{behavior_rules}"
                    "Be professional and use bullet points."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Request: {user_query}\n"
                    f"{memory_note}\n\n"
                    f"Dataset for Analysis:\n{data_snapshot}"
                )
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content