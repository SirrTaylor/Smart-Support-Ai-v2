# ai_command_center.py
import os
import re
import json
import pandas as pd
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
        if len(w) < 3 or w in STOPWORDS:
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

def build_cosmos_query(companies: list, terms: list):
    """
    Builds a Cosmos DB SQL API query from extracted keywords.
    Returns (query_string, parameters) tuple.
    """
    conditions = []
    parameters = []

    # Company conditions — OR between multiple companies
    if companies:
        company_conditions = " OR ".join([
            f"CONTAINS(UPPER(c.Company), @company{i})"
            for i in range(len(companies))
        ])
        conditions.append(f"({company_conditions})")
        for i, company in enumerate(companies):
            parameters.append({"name": f"@company{i}", "value": company.upper()})

    # Term conditions — each term searches across Company, sentiment, and Review
    for i, term in enumerate(terms):
        term_lower = term.lower()
        conditions.append(
            f"(CONTAINS(LOWER(c.Company), @term{i}) OR "
            f"CONTAINS(LOWER(c.sentiment), @term{i}) OR "
            f"CONTAINS(LOWER(c.Review), @term{i}))"
        )
        parameters.append({"name": f"@term{i}", "value": term_lower})

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    query = f"""
        SELECT 
            c.Company,
            c.sentiment,
            c.urgency,
            c.Rating,
            c.Review,
            c._ts
        FROM c
        WHERE {where_clause}
        OFFSET 0 LIMIT 50
    """
    return query, parameters


def query_cosmos_analysis(user_query: str):
    """
    Full pipeline: sanitize -> extract keywords -> query Cosmos DB.
    Returns a DataFrame or None if no results / invalid query.
    """
    # Step 1: Sanitize
    clean_query = sanitize_query_input(user_query)

    # Step 2: Extract keywords
    companies, terms = extract_keywords_from_query(clean_query)

    try:
        container = get_cosmos_container()

        # Step 3: Build and execute Cosmos query from extracted terms
        query, parameters = build_cosmos_query(companies, terms)

        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        # Step 4: Fallback retrieval for "real assistant" behavior.
        # If extraction/search misses, still fetch recent rows and optionally filter by sentiment intent.
        if not items:
            fallback_items = list(container.query_items(
                query="""
                    SELECT c.Company, c.sentiment, c.urgency, c.Rating, c.Review, c._ts
                    FROM c
                    ORDER BY c._ts DESC
                    OFFSET 0 LIMIT 100
                """,
                enable_cross_partition_query=True
            ))

            requested_sentiment = _extract_requested_sentiment(clean_query)
            if requested_sentiment:
                filtered = []
                for item in fallback_items:
                    sentiment_val = str(item.get("sentiment", "")).strip().lower()
                    if sentiment_val == requested_sentiment.lower():
                        filtered.append(item)
                items = filtered
            else:
                items = fallback_items

        if not items:
            return None

        # Step 5: Normalise into a DataFrame
        rows = []
        for item in items:
            rows.append({
                "company":    item.get("Company", ""),
                "sentiment":  item.get("sentiment", ""),
                "urgency":    item.get("urgency", ""),
                "rating":     item.get("Rating", ""),
                "review":     item.get("Review", ""),
                # Cosmos stores _ts as a Unix timestamp — convert to readable date
                "timestamp":  pd.to_datetime(item.get("_ts", 0), unit="s").strftime("%Y-%m-%d %H:%M")
            })

        return pd.DataFrame(rows)

    except Exception as e:
        raise RuntimeError(f"Cosmos DB query failed: {e}")


# --- GPT ANALYST ---

def run_analyst(user_query: str, results_df: pd.DataFrame) -> str:
    """
    Sends a capped, truncated data snapshot to GPT and returns the analyst report.
    """
    gpt_client = get_gpt_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # Cap rows and truncate review text to protect context window
    MAX_ROWS = 20
    display_df = results_df.head(MAX_ROWS).copy()
    display_df['review'] = display_df['review'].str[:200]
    data_snapshot = display_df[['company', 'sentiment', 'rating', 'review']].to_string(index=False)

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
                    "highlight keywords from reviews, end with one Strategic Insight. "
                    "Be professional and use bullet points."
                )
            },
            {
                "role": "user",
                "content": f"User Request: {user_query}\n\nDataset for Analysis:\n{data_snapshot}"
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content