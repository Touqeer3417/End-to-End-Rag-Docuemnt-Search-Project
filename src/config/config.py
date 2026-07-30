"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # For evaluation judge

    # LLM Models
    LLM_MODEL = "groq:llama-3.3-70b-versatile"
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")  # Evaluation ke liye

    # Document Processing
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    # LangSmith
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-document-intelligence")
    LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "true").lower() == "true"
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return init_chat_model(cls.LLM_MODEL)

    @classmethod
    def get_judge_llm(cls):
        """Initialize judge LLM for evaluation"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY missing in .env — required for evaluation judge")
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(f"openai:{cls.JUDGE_MODEL}")

    @classmethod
    def validate_tavily(cls):
        """Call once at startup — fails loud if Tavily key missing"""
        if not cls.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY missing in .env — required for Corrective RAG web search fallback"
            )
        os.environ["TAVILY_API_KEY"] = cls.TAVILY_API_KEY

