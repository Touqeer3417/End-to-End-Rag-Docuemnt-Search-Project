"""Configuration module for the RAG system."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def _get_setting(name: str, default: str = "") -> str:
    """Read a setting from environment/.env, then Streamlit Cloud secrets.

    Streamlit Community Cloud exposes pasted secrets through ``st.secrets``;
    it does not automatically place every secret into ``os.environ``.
    Import Streamlit lazily so the config can also be used in local scripts.
    """
    value = os.getenv(name)
    if value is not None and value != "":
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # Outside Streamlit there may be no secrets.toml; use the default.
        pass

    return default


# src/config/config.py -> project root is two parents above this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Central application configuration."""

    # ------------------------------------------------------------------
    # API KEYS
    # ------------------------------------------------------------------
    OPENAI_API_KEY = _get_setting("OPENAI_API_KEY", "")
    TAVILY_API_KEY = _get_setting("TAVILY_API_KEY", "")

    # ------------------------------------------------------------------
    # LOCAL CHROMADB
    # ------------------------------------------------------------------
    # No hosted database account or database API key is needed. Chroma writes
    # the index to this local folder. Set CHROMA_PERSIST_DIRECTORY to override.
    CHROMA_PERSIST_DIRECTORY = str(
        Path(
            _get_setting(
                "CHROMA_PERSIST_DIRECTORY",
                str(PROJECT_ROOT / ".chroma_db"),
            )
        ).expanduser()
    )

    CHROMA_COLLECTION_NAME = _get_setting(
        "CHROMA_COLLECTION_NAME",
        "documents_chroma",
    )

    # ------------------------------------------------------------------
    # OPENAI MODELS
    # ------------------------------------------------------------------
    LLM_MODEL = _get_setting(
        "OPENAI_MODEL",
        "gpt-5.6-luna",
    )

    EMBEDDING_MODEL = _get_setting(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    EMBEDDING_DIMENSIONS = int(
        _get_setting("OPENAI_EMBEDDING_DIMENSIONS", "1536")
    )

    JUDGE_MODEL = _get_setting(
        "JUDGE_MODEL",
        "gpt-5.6-luna",
    )

    # ------------------------------------------------------------------
    # DOCUMENT PROCESSING
    # ------------------------------------------------------------------
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    # ------------------------------------------------------------------
    # LANGSMITH
    # ------------------------------------------------------------------
    LANGSMITH_API_KEY = _get_setting("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = _get_setting(
        "LANGSMITH_PROJECT",
        "ai-document-intelligence",
    )
    LANGSMITH_TRACING_V2 = (
        _get_setting("LANGSMITH_TRACING_V2", "true").lower() == "true"
    )
    LANGSMITH_ENDPOINT = _get_setting(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    # ------------------------------------------------------------------
    # DEFAULT DOCUMENTS
    # ------------------------------------------------------------------
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # ------------------------------------------------------------------
    # MAIN LLM
    # ------------------------------------------------------------------
    @classmethod
    def get_llm(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY missing. Add it to your .env file or "
                "Streamlit secrets."
            )

        return ChatOpenAI(
            model=cls.LLM_MODEL,
            api_key=cls.OPENAI_API_KEY,
            streaming=True,
        )

    # ------------------------------------------------------------------
    # EVALUATION LLM
    # ------------------------------------------------------------------
    @classmethod
    def get_judge_llm(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY missing. Add it to your .env file or "
                "Streamlit secrets."
            )

        return ChatOpenAI(
            model=cls.JUDGE_MODEL,
            api_key=cls.OPENAI_API_KEY,
        )

    # ------------------------------------------------------------------
    # TAVILY
    # ------------------------------------------------------------------
    @classmethod
    def validate_tavily(cls):
        if not cls.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY missing — required for web search fallback."
            )
