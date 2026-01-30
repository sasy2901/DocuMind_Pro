import os
import base64
import logging
import tempfile
from dotenv import load_dotenv
from PIL import Image
from gtts import gTTS

# Streamlit & UI
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler

# LangChain & AI Core
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from groq import Groq

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="DocuMind Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI STYLING ---
# Custom CSS for dark mode optimization and component styling
st.markdown("""
<style>
    .stApp {background-color: #0E1117; color: #FAFAFA;}
    .stChatMessage {
        background-color: #1E2329;
        border: 1px solid #2B313A;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### **DocuMind Pro**")
    st.caption("Enterprise Agentic Workspace v1.0")
    st.divider()
    
    with st.expander("⚙️ **Model Configuration**", expanded=True):
        # Vision Model Selection
        # Using Llama 3.2 variants for multimodal inference
        vision_model_choice = st.selectbox(
            "👁️ Vision Model:",
            ("llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"),
            index=0 
        )
        
        # TTS Settings
        voice_option = st.selectbox(
            "🗣️ Audio Output:",
            ("🇺🇸 American", "🇮🇳 Indian", "🇬🇧 British", "🇦🇺 Australian"),
            index=1 
        )
        accent_map = {
            "🇺🇸 American": "com", "🇮🇳 Indian": "co.in",
            "🇬🇧 British": "co.uk", "🇦🇺 Australian": "com.au"
        }
        selected_tld = accent_map[voice_option]
        
        st.toggle("🔊 Enable TTS Response", value=True, key="autoplay")

    st.markdown("---")
    st.caption("© 2026 DocuMind AI Systems")

# --- CORE UTILITIES ---

def encode_image(image_path):
    """Encodes image to Base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@st.cache_resource
def load_agent_brain():
    """
    Initializes the RAG Retrieval Chain and Agent Tools.
    Cached resource to prevent re-initialization on every rerun.
    """
    if not API_KEY:
        st.error("❌ Configuration Error: GROQ_API_KEY missing in environment.")
        return None

    logging.info("Initializing Agent Core...")
    
    # 1. Vector Database Connection
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="vectorstore/db_chroma", embedding_function=embeddings)
    
    # 2. Retrieval Chain Configuration
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", groq_api_key=API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )

    # 3. Custom Knowledge Base Tool with Citation Support
    def ask_knowledge_base(query):
        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            sources = result["source_documents"]
            
            if not sources:
                return "Data not found in internal knowledge base."
            
            # Calculate heuristic confidence score based on retrieval density
            confidence_score = min(95, 70 + (len(sources) * 10)) 
            
            source_list = []
            for doc in sources:
                page = doc.metadata.get("page", "N/A")
                filename = doc.metadata.get("source", "Unknown File").split("/")[-1]
                source_list.append(f"📄 {filename} (Page {page})")
            
            unique_sources = "\n".join(list(set(source_list)))
            
            return (
                f"{answer}\n\n"
                f"---\n"
                f"📊 **Confidence:** {confidence_score}%\n"
                f"📚 **References:**\n{unique_sources}"
            )
            
        except Exception as e:
            logging.error(f"RAG Retrieval Error: {e}")
            return f"Retrieval Error: {str(e)}"

    # 4. Agent Tool Definitions
    search_tool = DuckDuckGoSearchRun()
    
    tools = [
        Tool(
            name="Internal Knowledge Base",
            func=ask_knowledge_base, 
            description="Primary source. Use for queries regarding uploaded documents."
        ),
        Tool(
            name="Web Search",
            func=search_tool.run,
            description="Secondary source. Use for current events or external facts."
        )
    ]

    # 5. Memory & Agent Initialization
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools,
        ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", groq_api_key=API_KEY),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, # Keep verbose for debugging in logs
        memory=memory, 
        max_iterations=5, 
        handle_parsing_errors=True 
    )
    
    logging.info("Agent System Online.")
    return agent

# --- MAIN APPLICATION ---

st.title("🧠 DocuMind Pro")
st.markdown("### Enterprise AI Workspace")

tab1, tab2 = st.tabs(["💬 **Agentic Chat**", "👁️ **Visual Analysis**"])

# === TAB 1: RAG AGENT ===
with tab1:
    agent_chain = load_agent_brain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Handling
    prompt = st.chat_input("Query knowledge base or search web...", key="main_chat")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if agent_chain:
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    response = agent_chain.run(input=prompt, callbacks=[st_callback])
                    st.markdown(response) 
                    st.caption("✅ Response validated by Multi-Agent System")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # TTS Playback
                    if st.session_state.autoplay:
                        try:
                            tts = gTTS(text=response, lang='en', tld=selected_tld)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                tts.save(tmp_file.name)
                                st.audio(tmp_file.name, format="audio/mp3")
                        except Exception as e:
                            logging.warning(f"TTS Error: {e}")
                            
                except Exception as e:
                    st.error(f"Agent Execution Failure: {str(e)}")

# === TAB 2: COMPUTER VISION ===
with tab2:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📤 Input Stream")
        uploaded_file = st.file_uploader("Upload Image Analysis Target", type=["jpg", "png", "jpeg"])
        
    with col2:
        st.markdown("#### 📊 Diagnostic Output")
        if uploaded_file:
            # 1. Render Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Analysis Target", use_container_width=True, channels="RGB")
            
            vision_prompt = st.text_area("Analysis Query:", value="Analyze this image for key entities, text data, and anomalies.")
            
            if st.button("🚀 Execute Visual Analysis", type="primary"):
                with st.spinner(f"Running inference on {vision_model_choice}..."):
                    try:
                        # 2. Pre-processing Pipeline
                        if image.mode in ("RGBA", "P"): 
                            image = image.convert("RGB")

                        # Optimization: Resize large images to reduce latency
                        max_width = 800
                        if image.width > max_width:
                            ratio = max_width / float(image.width)
                            new_height = int(float(image.height) * float(ratio))
                            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

                        # Encode for API
                        buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        image.save(buffered.name, format="JPEG", quality=85, optimize=True)
                        base64_image = encode_image(buffered.name)
                        
                        # 3. Inference Request
                        client = Groq(api_key=API_KEY)
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": vision_prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                    ],
                                }
                            ],
                            model=vision_model_choice, # Dynamically uses selected model
                        )
                        
                        # 4. Result Rendering
                        result = chat_completion.choices[0].message.content
                        st.success("Inference Complete")
                        st.markdown(result)
                        
                        # 5. Audio Feedback
                        if st.session_state.autoplay:
                            try:
                                tts = gTTS(text=result, lang='en', tld=selected_tld)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                    tts.save(tmp_file.name)
                                    st.audio(tmp_file.name, format="audio/mp3")
                            except Exception: pass      
                            
                    except Exception as e:
                        st.error(f"Vision Pipeline Error: {str(e)}")
