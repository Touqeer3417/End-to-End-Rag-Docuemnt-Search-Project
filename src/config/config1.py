"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the Agentic RAG system"""

    # =========================
    # API Keys
    # =========================
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For evaluation judge

    # =========================
    # LLM Models
    # =========================
    LLM_MODEL = os.getenv("LLM_MODEL", "groq:llama-3.3-70b-versatile")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

    # =========================
    # Document Processing
    # =========================
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # =========================
    # LangSmith
    # =========================
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-document-intelligence")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # Better boolean parsing
    LANGSMITH_TRACING_V2 = (
        os.getenv("LANGSMITH_TRACING_V2", "true")
        .strip()
        .lower()
        in ("true", "1", "yes", "on")
    )

    # =========================
    # Default URLs
    # =========================
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # =========================
    # Validation Helpers
    # =========================
    @classmethod
    def validate_groq(cls):
        """Validate Groq API key"""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY missing in .env — required for the main LLM."
            )

    @classmethod
    def validate_tavily(cls):
        """Validate Tavily API key"""
        if not cls.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY missing in .env — required for Corrective RAG web search fallback."
            )

    @classmethod
    def validate_openai(cls):
        """Validate OpenAI API key for evaluation"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY missing in .env — required for evaluation judge."
            )

    # =========================
    # LLM Factory Methods
    # =========================
    @classmethod
    def get_llm(cls):
        """Initialize and return the main LLM"""

        cls.validate_groq()

        # Ensure provider SDK can access the key
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY

        return init_chat_model(cls.LLM_MODEL)

    @classmethod
    def get_judge_llm(cls):
        """Initialize and return evaluation judge LLM"""

        cls.validate_openai()

        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY

        return init_chat_model(f"openai:{cls.JUDGE_MODEL}")

    # =========================
    # Startup Validation
    # =========================
    @classmethod
    def validate_startup(cls):
        """Run all required validations at application startup"""

        # Main LLM is mandatory
        cls.validate_groq()

        # Tavily is mandatory for Corrective RAG
        cls.validate_tavily()

        # OpenAI is optional unless evaluation is used
        if cls.OPENAI_API_KEY:
            cls.validate_openai()

        return True