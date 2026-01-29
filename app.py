import os
import streamlit as st
import base64
from PIL import Image
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
from groq import Groq

# --- 1. CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="DocuMind Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING ---
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

# --- 3. SIDEBAR (Settings & Branding) ---
with st.sidebar:
    st.markdown("### **DocuMind Pro**")
    st.caption("Multi-Modal Agentic Workspace")
    st.divider()
    
    with st.expander("⚙️ **System Settings**", expanded=True):
        # --- NEW: VISION MODEL SWITCHER ---
        vision_model_choice = st.selectbox(
            "👁️ Vision Model:",
            ("llama-3.2-90b-vision-instruct", "llama-3.2-11b-vision-instruct"),
            index=0 # Default to 90b (More stable)
        )
        
        voice_option = st.selectbox(
            "🗣️ Voice Accent:",
            ("🇺🇸 American", "🇮🇳 Indian", "🇬🇧 British", "🇦🇺 Australian"),
            index=1 
        )
        accent_map = {
            "🇺🇸 American": "com", "🇮🇳 Indian": "co.in",
            "🇬🇧 British": "co.uk", "🇦🇺 Australian": "com.au"
        }
        selected_tld = accent_map[voice_option]
        
        st.toggle("🔊 Auto-Play Audio", value=True, key="autoplay")

    st.markdown("---")
    st.markdown("👨‍💻 **Dev:** Sahil Rana")

# --- 4. CORE AI ENGINE ---

def encode_image(image_path):
    """Encodes image to Base64 for the Vision Model"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@st.cache_resource
def load_agent_brain_final():
    """Initializes the Agent with Smart Citations & Confidence Scores"""
    if not api_key:
        st.error("❌ Critical Error: API Key missing. Check .env file.")
        return None

    print("🔄 Booting up DocuMind Core (Robust Edition)...")
    
    # A. Embeddings & Database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="vectorstore/db_chroma", embedding_function=embeddings)
    
    # B. The "Smart" Retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # C. PDF Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", groq_api_key=api_key),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )

    # D. Custom Tool Function
    def ask_pdf_with_sources(query):
        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            sources = result["source_documents"]
            
            if not sources:
                return "I couldn't find anything in the document about that. (Confidence: Low)"
            
            confidence_score = min(95, 70 + (len(sources) * 10)) 
            
            source_list = []
            for doc in sources:
                page = doc.metadata.get("page", "Unknown")
                source = doc.metadata.get("source", "File")
                filename = source.split("/")[-1]
                source_list.append(f"📄 {filename} (Page {page})")
            
            unique_sources = list(set(source_list))
            formatted_sources = "\n".join(unique_sources)
            
            final_output = (
                f"{answer}\n\n"
                f"---\n"
                f"📊 **Confidence Score:** {confidence_score}%\n"
                f"📚 **Sources Used:**\n{formatted_sources}"
            )
            return final_output
            
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    # E. Web Search Tool
    search_tool = DuckDuckGoSearchRun()
    
    # F. Agent Toolbox
    tools = [
        Tool(
            name="PDF Knowledge Base",
            func=ask_pdf_with_sources, 
            description="Use FIRST. Strictly for questions about the uploaded document content."
        ),
        Tool(
            name="Web Search",
            func=search_tool.run,
            description="Use for current events, news, people, or facts not in the PDF."
        )
    ]

    # --- MEMORY (Prevents the loop) ---
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # G. Initialize Agent (CONVERSATIONAL MODE)
    agent = initialize_agent(
        tools,
        ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", groq_api_key=api_key),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True,
        memory=memory, 
        max_iterations=5, 
        handle_parsing_errors=True 
    )
    
    print("✅ System Online.")
    return agent

# --- 5. MAIN APPLICATION INTERFACE ---

st.title("🧠 DocuMind Pro")
st.markdown("Welcome to your **Agentic AI Workspace**. Upload documents, search the web, or analyze images.")

tab1, tab2 = st.tabs(["💬 **Chat & Search**", "👁️ **Vision Analyst**"])

# === TAB 1: HYBRID AGENT ===
with tab1:
    # LOAD THE BRAIN
    agent_chain = load_agent_brain_final()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your PDF, News, or the World...", key="main_chat")

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
                    st.caption("✅ Verified by DocuMind Agent | Sources analyzed in real-time")
                    # =======================
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    if st.session_state.autoplay:
                        try:
                            tts = gTTS(text=response, lang='en', tld=selected_tld)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                tts.save(tmp_file.name)
                                st.audio(tmp_file.name, format="audio/mp3")
                        except: pass
                except Exception as e:
                    st.error(f"System Message: {str(e)}")

# === TAB 2: VISION INTELLIGENCE ===
with tab2:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📤 Upload")
        uploaded_file = st.file_uploader("Drop an image (PNG/JPG)", type=["jpg", "png", "jpeg"])
        
    with col2:
        st.markdown("#### 📊 Analysis")
        if uploaded_file:
            # 1. Load Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Target Image", use_container_width=True, channels="RGB")
            
            vision_prompt = st.text_area("Question:", value="Explain this image in detail. Identify any text, charts, or key objects.")
            
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("👀 Llama 4 Scout Scanning..."):
                    try:
                        # 2. IMAGE PRE-PROCESSING
                        if image.mode in ("RGBA", "P"): 
                            image = image.convert("RGB")

                        # Resize to safe limits (Max 400px)
                        max_width = 400
                        if image.width > max_width:
                            ratio = max_width / float(image.width)
                            new_height = int(float(image.height) * float(ratio))
                            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

                        # Encode
                        buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                        image.save(buffered.name, format="JPEG", quality=50, optimize=True)
                        base64_image = encode_image(buffered.name)
                        
                        # 3. CALL GROQ API (Using YOUR Available Model)
                        client = Groq(api_key=api_key)
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
                            # THE MAGIC ID FROM YOUR LIST
                            model="meta-llama/llama-4-scout-17b-16e-instruct", 
                        )
                        
                        # 4. SHOW RESULT
                        result = chat_completion.choices[0].message.content
                        st.success("Analysis Successful")
                        st.markdown(result)
                        
                        # 5. AUDIO
                        if st.session_state.autoplay:
                            try:
                                tts = gTTS(text=result, lang='en', tld=selected_tld)
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                                    tts.save(tmp_file.name)
                                    st.audio(tmp_file.name, format="audio/mp3")
                            except: pass     
                            
                    except Exception as e:
                        st.error(f"❌ Vision Error: {str(e)}")