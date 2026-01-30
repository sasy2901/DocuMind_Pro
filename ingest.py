import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- LOGGING CONFIGURATION ---
# Production-grade logging to track pipeline execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
DATA_PATH = "data/"
DB_PATH = "vectorstore/db_chroma"

def create_vector_db():
    """
    Ingests PDF documents, generates embeddings, and persists them to ChromaDB.
    """
    logging.info("Initiating data ingestion pipeline...")

    # 1. LOAD DOCUMENTS
    # Using DirectoryLoader for scalable batch processing of unstructured PDF data.
    if not os.path.exists(DATA_PATH):
        logging.error(f"Data directory '{DATA_PATH}' not found.")
        return

    logging.info(f"Loading documents from: {DATA_PATH}")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        logging.warning("No PDF documents found in source directory. Aborting ingestion.")
        return

    logging.info(f"Successfully loaded {len(documents)} document pages.")

    # 2. TEXT SPLITTING strategy
    # Optimized chunk size (500) to balance granular retrieval with semantic coherence.
    # Overlap (50) ensures boundary continuity between chunks to prevent context loss.
    logging.info("Splitting documents into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    logging.info(f"Generated {len(texts)} chunks for processing.")

    # 3. EMBEDDING GENERATION
    # Leveraging 'all-MiniLM-L6-v2' for a high-efficiency balance of inference speed 
    # and retrieval accuracy (Dense Vector Representation).
    logging.info("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. VECTOR STORE PERSISTENCE
    # Persisting vectors to disk to enable low-latency retrieval without re-indexing.
    logging.info(f"Persisting vector store to {DB_PATH}...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    
    logging.info("Ingestion pipeline completed successfully. Vector store is ready.")

if __name__ == "__main__":
    create_vector_db()
