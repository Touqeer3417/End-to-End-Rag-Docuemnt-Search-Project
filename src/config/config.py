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

    
    LLM_MODEL = "groq:llama-3.3-70b-versatile"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

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
    def validate_tavily(cls):
        """Call once at startup — fails loud if Tavily key missing"""
        if not cls.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY missing in .env — required for Corrective RAG web search fallback"
            )
        os.environ["TAVILY_API_KEY"] = cls.TAVILY_API_KEY
