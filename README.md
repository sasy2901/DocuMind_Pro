# 🧠 DocuMind Pro: Enterprise-Grade Agentic RAG Workspace

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/sasy2901/DocuMind_Pro)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)
![AI Model](https://img.shields.io/badge/Model-Llama_3.2_Vision-purple)

**DocuMind Pro** is a Multi-Modal Agentic RAG (Retrieval-Augmented Generation) system designed to bridge the gap between static documents and real-time intelligence. Unlike traditional RAG, DocuMind uses an **Agentic Router** to dynamically decide between searching internal PDFs, browsing the live web, or analyzing complex visual data.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User Query] --> B{Agent Router}
    B -->|Technical/PDF| C[Vector Database (ChromaDB)]
    B -->|Real-Time Info| D[Web Search Tool (DuckDuckGo)]
    B -->|Image Analysis| E[Vision Model (Llama 3.2 Vision)]
    C --> F[Context Retrieval]
    D --> F
    E --> F
    F --> G[LLM Synthesis]
    G --> H[Final Response]
🚀 Key Features
🤖 Intelligent Agent Routing: Automatically detects user intent (Research vs. Data Retrieval vs. Image Analysis).

📚 Deep Document Understanding: Ingests and indexes 100+ page PDFs using Recursive Character Splitting.

👁️ Computer Vision Analyst: "Llama 3.2 Vision" interprets charts, diagrams, and physical scenes (e.g., machinery or sports).

🌐 Real-Time Web Connect: Fetches live stock prices, news, and market trends when internal data is insufficient.

🐳 Containerized Deployment: Fully Dockerized for seamless cloud deployment.

🛠️ Tech Stack
Orchestration: LangChain & Phidata

LLM Engine: Groq (Llama 3.2 90B)

Vector Store: ChromaDB (Persistent Storage)

Frontend: Streamlit (Custom CSS UI)

Embeddings: HuggingFace (All-MiniLM-L6-v2)
