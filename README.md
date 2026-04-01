# 💎 Smart Support AI: Analytics Pro
### Enterprise Cloud Sentiment Analysis & GPT-4 Command Center

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-support-v2.streamlit.app/)
[![Azure](https://img.shields.io/badge/Cloud-Azure-0089D6?style=flat&logo=microsoft-azure)](https://azure.microsoft.com/)

## Problem Statement
In modern e-commerce and SaaS, companies are flooded with thousands of customer reviews across multiple platforms. Manually categorizing these reviews is slow, prone to human bias, and makes it impossible to react to critical bugs in real-time.

**Smart Support AI** solves this by providing a high-speed, cloud-native pipeline that:
1.  **Automates Sentiment Analysis**: Instantly classifies feedback using Azure AI.
2.  **Centralizes Data**: Stores records in a globally scalable NoSQL database.
3.  **GPT-4 Insights**: Provides a natural language "Command Center" where managers can ask complex questions (e.g., "Summarize the technical issues from Android users in Poland") and get instant, data-driven answers.

---

## 🏗️ Architecture & Data Flow
The system follows a professional ETL (Extract, Transform, Load) and RAG (Retrieval-Augmented Generation) pattern:

**CSV Upload** → **Azure AI Language Service** (Sentiment Enrichment) → **Azure Cosmos DB** (NoSQL Storage) → **Azure OpenAI (GPT-4)** (Synthesis & Chat)

![Architecture](./architecture.png)

> **Pro Tip:** I used **Deterministic ID Logic** to ensure that re-uploading the same dataset updates existing records rather than creating duplicates—keeping the cloud database clean and cost-efficient.

---

## 🛠️ The "Enterprise" Stack
* **Frontend**: [Streamlit](https://streamlit.io/) (Premium Dashboard UI)
* **Language AI**: [Azure AI Language](https://azure.microsoft.com/en-us/products/ai-services/ai-language/) (Sentiment Analysis)
* **Brain**: [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/) (GPT-4 Turbo)
* **Database**: [Azure Cosmos DB](https://azure.microsoft.com/en-us/products/cosmos-db/) (NoSQL / Partitioned by Company)
* **Language**: Python 3.9+

---

## 🚀 Key Features
* **Multi-Company Support**: Filter and analyze data for different services (Netflix, Allegro, etc.) within one interface.
* **Smart Search**: AI-powered querying that understands context, not just keywords.
* **Cloud-Sync**: Real-time upserting to Azure cloud infrastructure.
* **Premium UI**: A clean, three-column dashboard optimized for desktop analytics.

---

## ⚙️ Setup & Installation

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   
3. **Configure Environment:**
Create a .env file based on the .env.example provided and add your Azure credentials.

3. **Run the App:**

   ```bash
   streamlit run app.py

👨‍💻 Author
Brendan Gobvu Cloud & AI Enthusiast | Based in Poland [![LinkedIn](https://img.shields.io/badge/LinkedIn-Brendan%20Gobvu-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/brendan-gobvu)

Built for the 2026 Enterprise AI Landscape.
