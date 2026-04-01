import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey

# 1. Setup and Environment
load_dotenv()

# The config file handles the colors; we handle the metadata here
st.set_page_config(
    page_title="Brendy Pro | Support AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Credentials
AZURE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AOAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")

DATABASE_NAME = "SupportAnalytics"
CONTAINER_NAME = "reviews"

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
        partition_key=PartitionKey(path="/Company"),
        offer_throughput=400
    )
    return container

def safe_enrich_and_upload(df, company_name):
    container = get_cosmos_container()
    azure_client = get_azure_client()
    
    df.columns = [c.strip() for c in df.columns]
    rev_col = next((c for c in df.columns if 'review' in c.lower()), None)
    
    if not rev_col:
        st.error("No review column found in the CSV!")
        return

    id_col = next((c for c in df.columns if 'id' in c.lower()), None)
    df = df.dropna(subset=[rev_col])
    
    st.info(f"🚀 Processing {len(df)} records for {company_name}...")
    progress_bar = st.progress(0)

    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row[rev_col])[:5000] 
        try:
            response = azure_client.analyze_sentiment(documents=[text])[0]
            sentiment = response.sentiment.capitalize() if not response.is_error else "Neutral"
        except:
            sentiment = "Neutral" 

        doc = row.to_dict()
        raw_id = str(row[id_col]).strip() if id_col and pd.notnull(row[id_col]) else str(i)
        doc['id'] = f"{company_name.upper()}_{raw_id}"
        doc['Company'] = company_name.upper()
        doc['sentiment'] = sentiment

        for col, val in doc.items():
            if isinstance(val, (pd.Timestamp, datetime)):
                doc[col] = val.isoformat()
        
        container.upsert_item(doc)
        progress_bar.progress((i + 1) / len(df))

# --- UI START: PREMIUM DASHBOARD ---
# Title and Subtitle
st.markdown("<h1 style='text-align: center;'>💎 Smart Support AI: Analytics Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enterprise Cloud Analytics | Azure Cosmos DB & GPT-4</p>", unsafe_allow_html=True)
st.divider()

# Create 3-Column Layout
col_left, col_mid, col_right = st.columns([2, 5, 3], gap="medium")

# --- LEFT COLUMN: SYSTEM STATUS ---
with col_left:
    with st.container(border=True):
        st.subheader("📊 System Status")
        st.write("🛰️ **Cloud:** Azure Cosmos DB")
        st.write("🧠 **Engine:** GPT-4 Turbo")
        st.write("⚡ **Mode:** Deterministic Sync")
        st.divider()
        st.markdown("**Active Environment:**")
        st.success("Connected to Azure")
        st.info("💡 **Tip:** Use IDs to prevent duplicates.")

# --- MIDDLE COLUMN: AI COMMAND CENTER ---
with col_mid:
    st.subheader("💬 AI Command Center")
    
    with st.container(border=True):
        query = st.text_input("Analyze cloud data...", placeholder="e.g., What was the rating for the review about the Android crash?")
        execute_btn = st.button("🚀 Execute Analysis", use_container_width=True)

        if execute_btn and query:
            container = get_cosmos_container()
            with st.spinner("Querying Cosmos DB..."):
                try:
                    all_items = list(container.read_all_items())
                    total_count = len(all_items)
                    
                    if total_count == 0:
                        st.warning("The database is currently empty.")
                    else:
                        # Logic to filter items based on query for better context management
                        if "negative" in query.lower():
                            items = [i for i in all_items if i.get('sentiment') == 'Negative']
                        elif "positive" in query.lower():
                            items = [i for i in all_items if i.get('sentiment') == 'Positive']
                        else:
                            items = all_items

                        pos = len([i for i in all_items if i.get('sentiment') == 'Positive'])
                        neg = len([i for i in all_items if i.get('sentiment') == 'Negative'])
                        neu = len([i for i in all_items if i.get('sentiment') == 'Neutral'])

                        gpt_client = get_gpt_client()
                        context_text = ""
                        for i in items[:15]:
                            review_text = i.get('review', i.get('Review', 'No text'))
                            rating = i.get('rating', i.get('Rating', 'N/A'))
                            date = i.get('date', i.get('Date', 'N/A'))
                            sentiment = i.get('sentiment', 'N/A')
                            context_text += f"DATA RECORD: [Review: {review_text}] | [Rating: {rating}] | [Date: {date}] | [Sentiment: {sentiment}]\n"

                        system_prompt = f"""You are a support data expert. 
                        Stats: Total: {total_count}, Pos: {pos}, Neg: {neg}, Neu: {neu}.
                        IMPORTANT: When asked about a 'Rating', look specifically at the [Rating] value. Be numerical and precise."""
                        
                        response = gpt_client.chat.completions.create(
                            model=AOAI_DEPLOYMENT,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Context Samples:\n{context_text}\n\nQuestion: {query}"}
                            ]
                        )
                        
                        st.markdown("---")
                        st.markdown(f"**AI Analyst:**\n\n{response.choices[0].message.content}")
                        st.info(f"📊 Live Stats: {pos} Positive | {neg} Negative | {neu} Neutral")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- RIGHT COLUMN: DATA SYNC ---
with col_right:
    with st.container(border=True):
        st.subheader("⬆️ Data Sync")
        
        with st.expander("📝 CSV Preparation Guide"):
            st.caption("Required columns in your csv: id, Review, Rating, Date")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file:
            sep_choice = st.selectbox("Separator", [",", ";", "Tab", "Auto-detect"])
            comp_name = st.text_input("Company Name", placeholder="e.g., Netflix").strip()
            
            st.divider()
            st.warning(f"Target: **{comp_name.upper() if comp_name else 'None'}**")
            
            is_confirmed = st.checkbox("Confirm details")

            if is_confirmed and comp_name:
                if st.button("🚀 Sync to Azure", use_container_width=True):
                    sep_map = {",": ",", ";": ";", "Tab": "\t", "Auto-detect": None}
                    current_sep = sep_map[sep_choice]
                    try:
                        df = pd.read_csv(uploaded_file, sep=current_sep if current_sep else None, 
                                        engine='python' if not current_sep else None)
                        safe_enrich_and_upload(df, comp_name)
                        st.success(f"Synced {comp_name}!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.caption("Complete fields to enable sync.")

st.divider()
st.markdown("<center><p style='color: gray;'>Built by Brendy Pro | 2026 | Enterprise Edition</p></center>", unsafe_allow_html=True)