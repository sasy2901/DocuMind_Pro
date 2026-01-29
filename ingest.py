import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/"  # The folder where you put your PDFs
DB_PATH = "vectorstore/db_chroma"  # Where the processed "Brain Memory" will be saved

def create_vector_db():
    print("🚀 Starting Ingestion Process...")

    # 1. LOAD THE DOCUMENTS
    # We use DirectoryLoader to find ALL files ending in .pdf inside the 'data' folder
    print(f"📂 Loading PDFs from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ No PDFs found! Please put a PDF in the 'data' folder first.")
        return

    print(f"✅ Loaded {len(documents)} pages from PDFs.")

    # 2. SPLIT TEXT INTO CHUNKS
    # AI can't read whole books at once. We cut them into "chunks" of 500 characters.
    # 'overlap' means we keep 50 characters from the previous chunk so context isn't lost.
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    print(f"🧩 Created {len(texts)} text chunks.")

    # 3. CREATE EMBEDDINGS & STORE
    # This turns text into "Vectors" (Numbers) that the AI can search mathematically.
    print("🧠 Creating Embeddings (This takes a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. SAVE TO DATABASE (ChromaDB)
    # This creates a real folder on your computer with the searchable data.
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    
    print(f"🎉 Success! Database saved to '{DB_PATH}'.")
    print("👉 You can now run 'streamlit run app.py' to chat with your docs!")

if __name__ == "__main__":
    create_vector_db()