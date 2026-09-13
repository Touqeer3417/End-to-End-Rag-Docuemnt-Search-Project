"""Qdrant vector store using OpenAI embeddings."""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from qdrant_client import QdrantClient, models

from src.config.config import Config


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Qdrant vector store using OpenAI embeddings.

    No HuggingFace.
    No BGE.
    No SentenceTransformers.
    No local reranker.
    """

    DENSE_NAME = "dense"

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "documents_openai",
        top_k: int = 4,
        fetch_k: int = 10,
        use_reranker: bool = False,
    ) -> None:

        self.collection_name = collection_name
        self.top_k = top_k
        self.fetch_k = fetch_k

        # =========================
        # QDRANT CLIENT
        # =========================
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
            check_compatibility=False,
        )

        # =========================
        # OPENAI EMBEDDINGS
        # =========================
        self.dense = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            dimensions=Config.EMBEDDING_DIMENSIONS,
            api_key=Config.OPENAI_API_KEY,
        )

        self.vectorstore: Optional[QdrantVectorStore] = None
        self.retriever: Optional[BaseRetriever] = None

    # ============================================================
    # PUBLIC METHODS
    # ============================================================

    def collection_exists(self) -> bool:
        """Check whether Qdrant collection exists."""

        try:
            return self.client.collection_exists(
                self.collection_name
            )

        except Exception as e:
            logger.error(
                "Error checking collection: %s",
                e
            )
            return False

    def has_documents(self) -> bool:
        """Check whether collection contains vectors."""

        try:
            if not self.collection_exists():
                return False

            info = self.client.get_collection(
                self.collection_name
            )

            return (info.points_count or 0) > 0

        except Exception as e:
            logger.error(
                "Error checking documents: %s",
                e
            )
            return False

    def get_document_count(self) -> int:
        """Return number of vectors stored."""

        try:
            if not self.collection_exists():
                return 0

            info = self.client.get_collection(
                self.collection_name
            )

            return info.points_count or 0

        except Exception as e:
            logger.error(
                "Error getting document count: %s",
                e
            )
            return 0

    def create_vectorstore(
        self,
        documents: List[Document],
        force: bool = False,
    ) -> None:
        """Create collection and index documents."""

        if not documents:
            raise ValueError(
                "Documents list is empty."
            )

        self._ensure_collection(
            recreate=force
        )

        logger.info(
            "Creating OpenAI embeddings for %d chunks...",
            len(documents),
        )

        self.vectorstore.add_documents(
            documents
        )

        self.retriever = (
            self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self.top_k
                }
            )
        )

        logger.info(
            "Indexed %d chunks into collection '%s'",
            len(documents),
            self.collection_name,
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve semantically relevant documents."""

        self._check_ready()

        k = k or self.top_k

        docs = self.vectorstore.similarity_search(
            query=query,
            k=k,
        )

        return docs

    def get_retriever(self) -> BaseRetriever:
        """Return LangChain retriever."""

        self._check_ready()

        if self.retriever is None:
            self.retriever = (
                self.vectorstore.as_retriever(
                    search_kwargs={
                        "k": self.top_k
                    }
                )
            )

        return self.retriever

    def load_existing(self) -> None:
        """Connect to existing OpenAI-embedding collection."""

        self._ensure_collection(
            recreate=False
        )

        self.retriever = (
            self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self.top_k
                }
            )
        )

        logger.info(
            "Connected to existing collection '%s'",
            self.collection_name,
        )

    # ============================================================
    # INTERNAL METHODS
    # ============================================================

    def _ensure_collection(
        self,
        recreate: bool = False,
    ) -> None:

        exists = self.client.collection_exists(
            self.collection_name
        )

        if exists and recreate:

            self.client.delete_collection(
                collection_name=self.collection_name
            )

            exists = False

            logger.info(
                "Deleted old collection '%s'",
                self.collection_name,
            )

        if not exists:

            self.client.create_collection(
                collection_name=self.collection_name,

                vectors_config={
                    self.DENSE_NAME: models.VectorParams(
                        size=Config.EMBEDDING_DIMENSIONS,
                        distance=models.Distance.COSINE,
                    )
                },
            )

            logger.info(
                "Created OpenAI vector collection '%s'",
                self.collection_name,
            )

        # Connect LangChain to Qdrant
        self.vectorstore = QdrantVectorStore(
            client=self.client,

            collection_name=self.collection_name,

            embedding=self.dense,

            retrieval_mode=RetrievalMode.DENSE,

            vector_name=self.DENSE_NAME,
        )

    def _check_ready(self) -> None:

        if self.vectorstore is None:
            raise ValueError(
                "Vector store is not initialized. "
                "Call create_vectorstore() or load_existing() first."
            )