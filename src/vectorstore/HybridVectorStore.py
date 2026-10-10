"""Persistent local ChromaDB vector store for the RAG application.

The historical module name is kept to avoid breaking existing imports. The
implementation no longer connects to Qdrant or needs a hosted vector database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from src.config.config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """Local persistent ChromaDB store using OpenAI dense embeddings.

    Chroma stores document chunks, embeddings, and metadata in the configured
    local directory. No Qdrant URL or Qdrant API key is used.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        top_k: int = 4,
        fetch_k: int = 10,
        use_reranker: bool = False,
    ) -> None:
        self.collection_name = (
            collection_name or Config.CHROMA_COLLECTION_NAME
        )
        self.persist_directory = str(
            Path(
                persist_directory or Config.CHROMA_PERSIST_DIRECTORY
            ).expanduser()
        )
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.use_reranker = use_reranker

        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is missing. ChromaDB is local and does not "
                "need a database API key, but this project still uses OpenAI "
                "for embeddings and its configured LLM. Add OPENAI_API_KEY "
                "to your .env file or Streamlit secrets."
            )

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # PersistentClient writes the collection to disk. It replaces the
        # hosted Qdrant client used by the previous implementation.
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.embedding_function = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            dimensions=Config.EMBEDDING_DIMENSIONS,
            api_key=Config.OPENAI_API_KEY,
        )

        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[BaseRetriever] = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Return True when the configured Chroma collection already exists."""
        try:
            collections = self.client.list_collections()
            names = {
                item if isinstance(item, str) else getattr(item, "name", "")
                for item in collections
            }
            return self.collection_name in names
        except Exception as exc:
            logger.exception("Could not list ChromaDB collections")
            raise RuntimeError(
                f"Could not check ChromaDB collection '{self.collection_name}' "
                f"in '{self.persist_directory}': {exc}"
            ) from exc

    def has_documents(self) -> bool:
        """Return True when the collection contains at least one document."""
        if not self.collection_exists():
            return False

        try:
            collection = self.client.get_collection(name=self.collection_name)
            return collection.count() > 0
        except Exception as exc:
            logger.exception("Could not count documents in ChromaDB")
            raise RuntimeError(
                f"Could not count documents in ChromaDB collection "
                f"'{self.collection_name}': {exc}"
            ) from exc

    def get_document_count(self) -> int:
        """Return the number of document chunks stored in ChromaDB."""
        if not self.collection_exists():
            return 0

        try:
            collection = self.client.get_collection(name=self.collection_name)
            return int(collection.count())
        except Exception as exc:
            logger.exception("Could not get ChromaDB document count")
            raise RuntimeError(
                f"Could not get document count for ChromaDB collection "
                f"'{self.collection_name}': {exc}"
            ) from exc

    def create_vectorstore(
        self,
        documents: List[Document],
        force: bool = False,
    ) -> None:
        """Create/connect to the collection and index document chunks.

        Set force=True to delete and rebuild only this collection. Other
        collections inside the same Chroma persistence directory are retained.
        """
        if not documents:
            raise ValueError("Documents list is empty; nothing can be indexed.")

        self._ensure_collection(recreate=force)
        assert self.vectorstore is not None

        logger.info(
            "Creating embeddings for %d chunks and adding them to ChromaDB...",
            len(documents),
        )
        self.vectorstore.add_documents(documents)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        logger.info(
            "Indexed %d chunks into ChromaDB collection '%s' at '%s'",
            len(documents),
            self.collection_name,
            self.persist_directory,
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve semantically similar document chunks."""
        self._check_ready()
        assert self.vectorstore is not None

        result_count = self.top_k if k is None else k
        return self.vectorstore.similarity_search(query=query, k=result_count)

    def get_retriever(self) -> BaseRetriever:
        """Return a LangChain retriever backed by the Chroma collection."""
        self._check_ready()
        assert self.vectorstore is not None

        if self.retriever is None:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
        return self.retriever

    def load_existing(self) -> None:
        """Attach to a previously indexed Chroma collection."""
        if not self.collection_exists():
            raise ValueError(
                f"ChromaDB collection '{self.collection_name}' does not exist "
                f"in '{self.persist_directory}'. Index documents first."
            )

        if not self.has_documents():
            raise ValueError(
                f"ChromaDB collection '{self.collection_name}' exists but is "
                "empty. Run document ingestion and indexing first."
            )

        self._ensure_collection(recreate=False)
        assert self.vectorstore is not None
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        logger.info(
            "Loaded ChromaDB collection '%s' from '%s' (%d chunks)",
            self.collection_name,
            self.persist_directory,
            self.get_document_count(),
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Create a Chroma collection if missing, then attach LangChain to it."""
        exists = self.collection_exists()

        if exists and recreate:
            self.client.delete_collection(name=self.collection_name)
            exists = False
            logger.info("Deleted existing ChromaDB collection '%s'", self.collection_name)

        # Chroma's HNSW space is configured to cosine similarity to match the
        # distance metric used by the former Qdrant collection.
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
        )

        if not exists:
            logger.info(
                "Created ChromaDB collection '%s' in '%s'",
                self.collection_name,
                self.persist_directory,
            )

    def _check_ready(self) -> None:
        if self.vectorstore is None:
            raise ValueError(
                "ChromaDB vector store is not initialized. Call "
                "create_vectorstore() or load_existing() first."
            )
