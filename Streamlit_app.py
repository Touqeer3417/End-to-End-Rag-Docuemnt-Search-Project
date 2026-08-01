import os
import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorStore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from langsmith import wrappers
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
os.environ["USER_AGENT"] = "Touqeer-RAG-App/1.0" 

# ═══════════════════════════
# NEW IMPORTS FOR PDF LOADING
# ═══════════════════════════
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 

sys.path.append(str(Path(__file__).parent))



def load_css():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        script_dir / "assets" / "style.css",      # Same folder
        script_dir / "style.css",                  # Root mein
        script_dir / ".." / "assets" / "style.css", # Parent folder
        Path("assets") / "style.css",               # CWD se
        Path(os.getcwd()) / "assets" / "style.css", # Absolute CWD
    ]
    
    for path in possible_paths:
        if path.resolve().exists():
            css_content = path.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
    
    st.warning("⚠️ CSS file not found! Make sure 'assets/style.css' exists.")



st.set_page_config(
    page_title=" AI Document Intelligence",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load external CSS
load_css()


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'total_chunks' not in st.session_state:
        st.session_state.total_chunks = 0



@st.cache_resource(show_spinner=False)
def initialize_rag():
    try:
        llm = Config.get_llm()

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        documents = doc_processor.process_documents(
            urls=Config.DEFAULT_URLS,
            data_folder="data"
        )

        vector_store = VectorStore(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            collection_name=Config.QDRANT_COLLECTION_NAME
        )
     
        vector_store.create_vectorstore(documents)

        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0


def render_particles():
    particles_html = ""
    for i in range(20):
        left = (i * 5.3) % 100
        top = (i * 7.7) % 100
        delay = (i * 0.4) % 6
        duration = 5 + (i % 5)
        size = 2 + (i % 4)
        opacity = 0.2 + (i % 3) * 0.15
        particles_html += f'''
            <div class="particle" style="
                left:{left}%;
                top:{top}%;
                width:{size}px;
                height:{size}px;
                animation-delay:{delay}s;
                animation-duration:{duration}s;
                opacity:{opacity};
                animation: float {duration}s ease-in-out infinite;
            "></div>
        '''
    st.markdown(particles_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    init_session_state()
    render_particles()

    # ─── Hero Section ───
    st.markdown("""
        <div class="hero-container">
            <div class="hero-emoji">🧠</div>
            <div class="hero-title">Intelligent Document Intelligence Engine</div>
          
        </div>
    """, unsafe_allow_html=True)

    # ─── Initialize ───
    if not st.session_state.initialized:
        with st.spinner("⚡ Initializing Neural Engine..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.total_chunks = num_chunks
                st.session_state.initialized = True
                st.success(f"🚀 Engine Ready — {num_chunks:,} document chunks indexed")
                time.sleep(0.5)
                st.rerun()

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ─── Search Form ───
    with st.form("search_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a Question  ",
            placeholder="Ask anything about your documents...",
            label_visibility="collapsed"
        )
        submit = st.form_submit_button("🔍 Search Documents")

    # ─── Process Search ───
    if submit and question and st.session_state.rag_system:
        start_time = time.time()
        answer_placeholder = st.empty()
        answer_text = ""

        run_config = RunnableConfig(
            metadata={
                "environment": "production",
                "app_version": "1.0.0",
                "user_id": st.session_state.get("user_id", "anonymous"),
                "session_id": st.session_state.get("session_id", "default"),
            }
        )

        # Streaming answer
        for msg_chunk, metadata in st.session_state.rag_system.stream(question, config=run_config):
            if metadata.get("langgraph_node") == "responder":
                token = msg_chunk.content
                if token:
                    answer_text += token
                    answer_placeholder.markdown(f"""
                        <div class="answer-wrapper">
                            <div class="answer-header">
                                <span class="answer-icon">✨</span>
                                <span class="answer-label">Generated Answer</span>
                            </div>
                            <div class="answer-text">{answer_text}▌</div>
                        </div>
                    """, unsafe_allow_html=True)

        elapsed_time = time.time() - start_time

        # Final answer with meta
        answer_placeholder.markdown(f"""
            <div class="answer-wrapper">
                <div class="answer-header">
                    <span class="answer-icon">✨</span>
                    <span class="answer-label">Generated Answer</span>
                </div>
                <div class="answer-text">{answer_text}</div>
                <div class="meta-bar">
                    <div class="meta-time">⏱️ Generated in {elapsed_time:.2f}s</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Save to history
        st.session_state.history.append({
            'question': question,
            'answer': answer_text,
            'time': elapsed_time,
            'timestamp': datetime.now().strftime("%H:%M")
        })

    elif submit and not st.session_state.rag_system:
        st.error("⚠️ System not initialized. Please refresh the page.")

    # ─── History Section ───
    if st.session_state.history:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="history-title">📜 Recent Queries</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="text-align:right;">', unsafe_allow_html=True)
            if st.button("🗑️ Clear", key="clear_history", help="Clear all history"):
                st.session_state.history = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
                <div class="history-card">
                    <div class="history-question">❓ {item['question']}</div>
                    <div class="history-answer">{item['answer']}</div>
                    <div class="history-footer">
                        <div class="history-time">🕐 {item.get('timestamp', 'Now')}</div>
                        <div class="history-speed">⚡ {item['time']:.2f}s</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔮</div>
                <div class="empty-text">Your search history will appear here</div>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()