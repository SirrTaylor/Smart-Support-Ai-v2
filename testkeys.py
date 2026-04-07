from azure.cosmos import CosmosClient
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


def test_connection():
    try:
        client = CosmosClient(os.getenv("AZURE_COSMOS_ENDPOINT"), os.getenv("AZURE_COSMOS_KEY"))
        # This just fetches account info to see if the key works
        properties = client.get_database_account()
        return "✅ Connection Successful!"
    except Exception as e:
        return f"❌ Connection Failed: {str(e)}"

st.write(test_connection())