"""Configuration module for RAG system."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class Config:
    """Configuration class for RAG system."""

    # =========================
    # API KEYS
    # =========================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # =========================
    # QDRANT
    # =========================
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

    # IMPORTANT:
    # New collection because old collection contains BGE embeddings
    QDRANT_COLLECTION_NAME = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "documents_openai"
    )

    # =========================
    # OPENAI MODELS
    # =========================
    LLM_MODEL = os.getenv(
        "OPENAI_MODEL",
        "gpt-5.6-luna"
    )

    EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small"
    )

    EMBEDDING_DIMENSIONS = int(
        os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")
    )

    # Evaluation model
    JUDGE_MODEL = os.getenv(
        "JUDGE_MODEL",
        "gpt-5.6-luna"
    )

    # =========================
    # DOCUMENT PROCESSING
    # =========================
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    # =========================
    # LANGSMITH
    # =========================
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "ai-document-intelligence"
    )

    LANGSMITH_TRACING_V2 = (
        os.getenv("LANGSMITH_TRACING_V2", "true").lower() == "true"
    )

    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com"
    )

    # =========================
    # DEFAULT DOCUMENTS
    # =========================
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # =========================
    # MAIN LLM
    # =========================
    @classmethod
    def get_llm(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY missing. Add it to your .env file."
            )

        return ChatOpenAI(
            model=cls.LLM_MODEL,
            api_key=cls.OPENAI_API_KEY,
            streaming=True,
        )

    # =========================
    # EVALUATION LLM
    # =========================
    @classmethod
    def get_judge_llm(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY missing. Add it to your .env file."
            )

        return ChatOpenAI(
            model=cls.JUDGE_MODEL,
            api_key=cls.OPENAI_API_KEY,
        )

    # =========================
    # TAVILY
    # =========================
    @classmethod
    def validate_tavily(cls):
        if not cls.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY missing — required for web search fallback."
            )