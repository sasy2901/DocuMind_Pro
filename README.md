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
    B --> C[Vector Database]
    B --> D[Web Search Tool]
    B --> E[Vision Model]
    C --> F[Context Retrieval]
    D --> F
    E --> F
    F --> G[LLM Synthesis]
    G --> H[Final Response]
I know this is annoying, but we are going to fix it once and for all.

The reason it failed is that GitHub's diagram tool is very sensitive to symbols like ( ) or /. Even a tiny typo breaks the whole picture.

We are going to use the "Bulletproof Version" (Safe Mode). I have removed all the confusing symbols so it cannot fail.

Step 1: Open the Editor
Go back to your GitHub README file.

Click the Pencil Icon (✏️) to edit.

Delete everything. (Make it empty).

Step 2: Paste this SAFE Code
Copy this exact block. I simplified the diagram labels to ensure it renders perfectly.

Markdown
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
    B --> C[Vector Database]
    B --> D[Web Search Tool]
    B --> E[Vision Model]
    C --> F[Context Retrieval]
    D --> F
    E --> F
    F --> G[LLM Synthesis]
    G --> H[Final Response]
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
