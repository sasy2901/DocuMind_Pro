# 🧠 DocuMind Pro: Enterprise-Grade Agentic RAG Workspace

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/sasy2901/DocuMind_Pro)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![AI Model](https://img.shields.io/badge/Model-Llama_3.2_Vision-purple)

**DocuMind Pro** is a Multi-Modal Agentic RAG system. It uses an **Agentic Router** to dynamically decide between searching internal PDFs, browsing the live web, or analyzing complex visual data.

---

## 🏗️ System Architecture

```mermaid
graph TD
    UserQuery[User Query] --> Router[Agent Router]
    Router --> ChromaDB[Vector Database]
    Router --> WebSearch[Web Search Tool]
    Router --> Vision[Vision Model]
    ChromaDB --> Context[Context Retrieval]
    WebSearch --> Context
    Vision --> Context
    Context --> LLM[LLM Synthesis]
    LLM --> Response[Final Response]
```
🚀 Key Features
Intelligent Agent Routing: Automatically detects user intent.

Deep Document Understanding: Ingests and indexes 100+ page PDFs.

Computer Vision Analyst: Interprets charts, diagrams, and physical scenes.

Real-Time Web Connect: Fetches live stock prices and news.

Containerized Deployment: Fully Dockerized for cloud.

🛠️ Tech Stack
Orchestration: LangChain and Phidata

LLM Engine: Groq (Llama 3.2 90B)

Vector Store: ChromaDB

Frontend: Streamlit

Embeddings: HuggingFace

💻 Installation and Usage
Clone the repository:

Bash
git clone [https://github.com/sasy2901/DocuMind_Pro.git](https://github.com/sasy2901/DocuMind_Pro.git)
Install dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
streamlit run app.py
