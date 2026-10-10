import os
import sys
import time
import html
import textwrap
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.HybridVectorStore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()

os.environ["USER_AGENT"] = "ai-document-intelligence/1.0"

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# HTML HELPERS
# ============================================================

def clean_html(content: str) -> str:
    """
    Convert multiline indented HTML into one continuous line.

    This prevents Streamlit Markdown from treating nested,
    indented HTML as a Markdown code block.
    """

    if not content:
        return ""

    dedented = textwrap.dedent(content)

    return "".join(
        line.strip()
        for line in dedented.splitlines()
        if line.strip()
    )


def render_html(content: str) -> None:
    """
    Safely render custom HTML in Streamlit.
    """

    st.markdown(
        clean_html(content),
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

def load_css() -> None:

    possible_paths = [
        PROJECT_ROOT / "assets" / "style.css",
        PROJECT_ROOT / "style.css",
        Path("assets") / "style.css",
    ]

    for css_path in possible_paths:

        if css_path.exists():

            css = css_path.read_text(
                encoding="utf-8"
            )

            st.markdown(
                f"<style>{css}</style>",
                unsafe_allow_html=True,
            )

            return

    st.warning(
        "⚠️ CSS file not found. "
        "Expected: assets/style.css"
    )


load_css()


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:

    default_values = {

        "rag_system": None,

        "initialized": False,

        "history": [],

        "total_chunks": 0,

        "last_question": "",

        "last_answer": "",

        "last_response_time": 0.0,
    }

    for key, default_value in default_values.items():

        if key not in st.session_state:

            st.session_state[key] = (
                default_value
            )


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    # --------------------------------------------------------
    # 1. LLM
    # --------------------------------------------------------

    llm = Config.get_llm()

    # --------------------------------------------------------
    # 2. VECTOR STORE
    # --------------------------------------------------------

    vector_store = VectorStore(
        collection_name=Config.CHROMA_COLLECTION_NAME,
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
    )

    # --------------------------------------------------------
    # 3. LOAD EXISTING DATA
    # --------------------------------------------------------

    if (
        vector_store.collection_exists()
        and vector_store.has_documents()
    ):

        vector_store.load_existing()

    # --------------------------------------------------------
    # 4. FIRST-TIME INDEXING
    # --------------------------------------------------------

    else:

        document_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        documents = (
            document_processor.process_documents(
                urls=Config.DEFAULT_URLS,
                data_folder="data",
            )
        )

        if not documents:

            raise ValueError(
                "No documents were loaded for indexing."
            )

        vector_store.create_vectorstore(
            documents
        )

    # --------------------------------------------------------
    # 5. DOCUMENT COUNT
    # --------------------------------------------------------

    doc_count = (
        vector_store.get_document_count()
    )

    # --------------------------------------------------------
    # 6. BUILD RAG GRAPH
    # --------------------------------------------------------

    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm,
    )

    graph_builder.build()

    return graph_builder, doc_count


# ============================================================
# PARTICLES
# ============================================================

def render_particles() -> None:

    particle_elements = []

    for i in range(22):

        left = (i * 7.1) % 100

        top = (i * 11.3) % 100

        delay = (i * 0.35) % 8

        duration = 7 + (i % 6)

        size = 2 + (i % 3)

        particle_elements.append(

            (
                '<span class="particle" '
                f'style="left:{left}%;'
                f'top:{top}%;'
                f'width:{size}px;'
                f'height:{size}px;'
                f'animation-delay:{delay}s;'
                f'animation-duration:{duration}s;">'
                '</span>'
            )
        )

    particles_html = (
        '<div class="particle-layer">'
        + "".join(particle_elements)
        + "</div>"
    )

    st.markdown(
        particles_html,
        unsafe_allow_html=True,
    )


# ============================================================
# STREAM CONTENT NORMALIZATION
# ============================================================

def chunk_to_text(content) -> str:

    if content is None:

        return ""

    if isinstance(content, str):

        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                text_value = item.get("text")

                if text_value:

                    parts.append(
                        str(text_value)
                    )

        return "".join(parts)

    return str(content)


# ============================================================
# HEADER
# ============================================================

def render_header() -> None:

    if st.session_state.initialized:

        status_text = "Ready"

        status_class = "ready"

    else:

        status_text = "Starting"

        status_class = "starting"

    render_html(
        f"""
        <div class="app-header">

            <div class="brand-area">

                <div class="brand-icon">
                    🧠
                </div>

                <div class="brand-copy">

                    <div class="brand-title">
                        AI Document Intelligence
                    </div>

                    <div class="brand-description">
                        Search, understand and reason across your documents
                    </div>

                </div>

            </div>

            <div class="engine-pill">

                <span class="status-dot {status_class}"></span>

                <span>
                    {status_text}
                </span>

            </div>

        </div>
        """
    )


# ============================================================
# INTRO SECTION
# ============================================================

def render_intro() -> None:

    render_html(
        """
        <div class="workspace-intro">

            <div class="intro-eyebrow">
                AI KNOWLEDGE WORKSPACE
            </div>

            <h1 class="workspace-title">
                Ask your documents.
                <span>
                    Get intelligent answers.
                </span>
            </h1>

            <p class="workspace-description">
                Search your indexed knowledge using AI-powered
                semantic retrieval and intelligent reasoning.
            </p>

        </div>
        """
    )


# ============================================================
# STARTUP UI
# ============================================================

def render_startup(message: str) -> None:

    safe_message = html.escape(message)

    render_html(
        f"""
        <div class="startup-card">

            <div class="startup-spinner"></div>

            <div>

                <div class="startup-title">
                    Starting AI Engine
                </div>

                <div class="startup-subtitle">
                    {safe_message}
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# ANSWER HEADER
# ============================================================

def render_answer_header() -> None:

    render_html(
        """
        <div class="answer-heading-row">

            <div>

                <div class="answer-eyebrow">
                    AI RESPONSE
                </div>

                <div class="answer-heading">

                    <span class="sparkle-icon">
                        ✦
                    </span>

                    Generated Answer

                </div>

            </div>

            <div class="completed-pill">

                <span></span>

                Completed

            </div>

        </div>
        """
    )


# ============================================================
# RESPONSE METRICS
# ============================================================

def render_response_metrics(
    response_time: float,
) -> None:

    total_chunks = (
        st.session_state.total_chunks
    )

    render_html(
        f"""
        <div class="response-metrics">

         


        </div>
        """
    )


# ============================================================
# RECENT SEARCHES
# ============================================================

def render_recent_searches() -> None:

    if not st.session_state.history:

        return

    render_html(
        """
        <div class="recent-section-heading">

            <div class="recent-eyebrow">
                YOUR ACTIVITY
            </div>

            <div class="recent-title">
                Recent Searches
            </div>

        </div>
        """
    )

    recent_items = list(
        reversed(
            st.session_state.history[-3:]
        )
    )

    columns = st.columns(
        len(recent_items),
        gap="medium",
    )

    for column, item in zip(
        columns,
        recent_items,
    ):

        safe_question = html.escape(
            item["question"]
        )

        timestamp = html.escape(
            item.get(
                "timestamp",
                "Now",
            )
        )

        with column:

            render_html(
                f"""
                <div class="recent-search-card">

                    <div class="recent-card-top">

                        <span class="recent-icon">
                            ↗
                        </span>

                        <span class="recent-time">
                            {timestamp}
                        </span>

                    </div>

                    <div class="recent-question">
                        {safe_question}
                    </div>

                    <div class="recent-card-footer">

                        <span>
                            AI Search
                        </span>

                        <span class="recent-speed">
                            ⚡ {item["time"]:.2f}s
                        </span>

                    </div>

                </div>
                """
            )

    empty_col, clear_col = st.columns(
        [6, 1]
    )

    with clear_col:

        if st.button(
            "Clear history",
            key="clear_history",
            use_container_width=True,
        ):

            st.session_state.history = []

            st.session_state.last_question = ""

            st.session_state.last_answer = ""

            st.session_state.last_response_time = 0.0

            st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

def render_empty_state() -> None:

    render_html(
        """
        <div class="empty-workspace">

            <div class="empty-visual">
                ✦
            </div>

            <div class="empty-heading">
                Your knowledge is ready
            </div>

            <div class="empty-description">
                Ask a question above to begin exploring
                your indexed documents.
            </div>

        </div>
        """
    )


# ============================================================
# RUN QUERY
# ============================================================

def run_query(
    question: str,
) -> None:

    start_time = time.time()

    answer_text = ""

    # Store latest question
    st.session_state.last_question = (
        question
    )

    # --------------------------------------------------------
    # ANSWER CONTAINER
    # --------------------------------------------------------

    with st.container(
        key="answer_panel"
    ):

        render_answer_header()

        answer_placeholder = st.empty()

        # ----------------------------------------------------
        # RUN CONFIG
        # ----------------------------------------------------

        run_config = RunnableConfig(

            metadata={

                "environment":
                    "production",

                "app_version":
                    "2.0.0",

                "user_id":
                    st.session_state.get(
                        "user_id",
                        "anonymous",
                    ),

                "session_id":
                    st.session_state.get(
                        "session_id",
                        "default",
                    ),
            }
        )

        # ----------------------------------------------------
        # STREAM ANSWER
        # ----------------------------------------------------

        try:

            response_stream = (
                st.session_state
                .rag_system
                .stream(
                    question,
                    config=run_config,
                )
            )

            for (
                message_chunk,
                metadata,
            ) in response_stream:

                node_name = (
                    metadata.get(
                        "langgraph_node"
                    )
                )

                if node_name != "responder":

                    continue

                token = chunk_to_text(
                    getattr(
                        message_chunk,
                        "content",
                        "",
                    )
                )

                if not token:

                    continue

                answer_text += token

                # Render Markdown normally.
                # Do NOT use unsafe HTML here.
                answer_placeholder.markdown(
                    answer_text + " ▌"
                )

        except Exception as exc:

            st.error(
                f"❌ Search failed: {str(exc)}"
            )

            return

        # ----------------------------------------------------
        # FINISHED
        # ----------------------------------------------------

        elapsed_time = (
            time.time()
            - start_time
        )

        if not answer_text.strip():

            answer_text = (
                "No answer was generated. "
                "Please try another question."
            )

        answer_placeholder.markdown(
            answer_text
        )

        render_response_metrics(
            elapsed_time
        )

    # --------------------------------------------------------
    # SAVE LATEST RESPONSE
    # --------------------------------------------------------

    st.session_state.last_answer = (
        answer_text
    )

    st.session_state.last_response_time = (
        elapsed_time
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    st.session_state.history.append(

        {

            "question":
                question,

            "answer":
                answer_text,

            "time":
                elapsed_time,

            "timestamp":
                datetime.now().strftime(
                    "%H:%M"
                ),
        }
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    init_session_state()

    # Background particles
    render_particles()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    render_header()

    # --------------------------------------------------------
    # INITIALIZE SYSTEM
    # --------------------------------------------------------

    if not st.session_state.initialized:

        startup_placeholder = (
            st.empty()
        )

        startup_placeholder.markdown(
            clean_html(
                """
                <div class="startup-card">

                    <div class="startup-spinner"></div>

                    <div>

                        <div class="startup-title">
                            Starting AI Engine
                        </div>

                        <div class="startup-subtitle">
                            Connecting to your knowledge base...
                        </div>

                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        try:

            rag_system, num_chunks = (
                initialize_rag()
            )

        except Exception as exc:

            startup_placeholder.empty()

            st.error(
                f"❌ Initialization failed: {str(exc)}"
            )

            st.exception(exc)

            return

        startup_placeholder.empty()

        if rag_system is None:

            st.error(
                "RAG system could not be initialized."
            )

            return

        st.session_state.rag_system = (
            rag_system
        )

        st.session_state.total_chunks = (
            num_chunks
        )

        st.session_state.initialized = True

        # Required so header changes:
        # Starting -> Ready
        st.rerun()

    # --------------------------------------------------------
    # INTRO
    # --------------------------------------------------------

    render_intro()

    # --------------------------------------------------------
    # SEARCH FORM
    # --------------------------------------------------------

    with st.form(
        "search_form",
        clear_on_submit=False,
    ):

        input_column, button_column = (
            st.columns(
                [6.5, 1.35],
                vertical_alignment="bottom",
                gap="small",
            )
        )

        with input_column:

            question = st.text_input(

                "Search your knowledge",

                placeholder=(
                    "Ask anything about "
                    "your documents..."
                ),

                label_visibility="collapsed",
            )

        with button_column:

            submitted = (
                st.form_submit_button(
                    "Search  →",
                    use_container_width=True,
                )
            )

    # --------------------------------------------------------
    # NEW SEARCH
    # --------------------------------------------------------

    if submitted:

        clean_question = (
            question.strip()
        )

        if not clean_question:

            st.warning(
                "⚠️ Please enter a question first."
            )

        elif not st.session_state.rag_system:

            st.error(
                "❌ AI engine is not initialized."
            )

        else:

            run_query(
                clean_question
            )

    # --------------------------------------------------------
    # KEEP LAST ANSWER VISIBLE AFTER RERUN
    # --------------------------------------------------------

    elif st.session_state.last_answer:

        with st.container(
            key="answer_panel"
        ):

            render_answer_header()

            st.markdown(
                st.session_state.last_answer
            )

            render_response_metrics(
                st.session_state
                .last_response_time
            )

    # --------------------------------------------------------
    # RECENT SEARCHES
    # --------------------------------------------------------

    render_recent_searches()

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    if (
        st.session_state.initialized
        and not st.session_state.last_answer
        and not st.session_state.history
    ):

        render_empty_state()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    main()