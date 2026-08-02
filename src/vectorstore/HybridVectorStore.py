"""Vector store with hybrid search (dense + BM25) and reranking."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class _RerankRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper."""
    store: "VectorStore"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self.store.retrieve(query)


class VectorStore:
    """Hybrid (dense + sparse) Qdrant vector store with optional reranking."""
   
    SPARSE_MODEL = "Qdrant/bm25"
    DENSE_NAME = "dense"
    SPARSE_NAME = "sparse"

    DENSE_MODEL = "BAAI/bge-m3"
    DENSE_SIZE = 1024      

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "documents",
        top_k: int = 4,
        fetch_k: int = 20,
        use_reranker: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)

        # Qdrant client
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
            check_compatibility=False
        )

        # Embeddings
        self.dense = HuggingFaceEmbeddings(
            model_name=self.DENSE_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.sparse = FastEmbedSparse(model_name=self.SPARSE_MODEL)

        # Reranker
        self.reranker = self._load_reranker() if use_reranker else None

        # Internal state
        self.vectorstore: Optional[QdrantVectorStore] = None
        self.retriever: Optional[BaseRetriever] = None

    # ==================== PUBLIC API ====================

    def collection_exists(self) -> bool:
        """Check karo ke Qdrant mein collection already hai ya nahi"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            return False

    def has_documents(self) -> bool:
        """Check karo ke collection mein data hai ya nahi"""
        try:
            if not self.collection_exists():
                return False
            info = self.client.get_collection(self.collection_name)
            return info.points_count > 0
        except Exception as e:
            logger.error(f"Error checking documents: {e}")
            return False

    def get_document_count(self) -> int:
        """Kitne documents indexed hain"""
        try:
            if not self.collection_exists():
                return 0
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def create_vectorstore(
        self, documents: List[Document], force: bool = False  # 👈 Default False
    ) -> None:
        """
        Collection create karo (agar nahi hai), documents index karo,
        aur retriever setup karo.
        """
        if not documents:
            raise ValueError("Documents list khali hai!")

        self._ensure_collection(recreate=force)

        # Documents add karo
        self.vectorstore.add_documents(documents)
        
        # Retriever ready karo
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        logger.info("Indexed %d documents into '%s'", len(documents), self.collection_name)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        self._check_ready()
        k = k or self.top_k
        fetch = self.fetch_k if self.reranker else k

        docs = self.vectorstore.similarity_search(query=query, k=fetch)

        if self.reranker:
            docs = self._rerank(query, docs, k)

        return docs

    def get_retriever(self) -> BaseRetriever:
        self._check_ready()
        if self.retriever is None:
            self.retriever = _RerankRetriever(store=self)
        return self.retriever

    def load_existing(self) -> None:
        self._ensure_collection(recreate=False)
        logger.info("Connected to existing collection '%s'", self.collection_name)
                
      

    # ==================== INTERNALS ====================

    def _ensure_collection(self, recreate: bool) -> None:
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
            logger.info("Old collection '%s' deleted", self.collection_name)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.DENSE_NAME: models.VectorParams(
                        size=self.DENSE_SIZE, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
            logger.info("Created collection '%s'", self.collection_name)

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense,
            sparse_embedding=self.sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=self.DENSE_NAME,
            sparse_vector_name=self.SPARSE_NAME,
        )

    def _load_reranker(self):
        if os.getenv("COHERE_API_KEY"):
            try:
                import cohere
                client = cohere.Client(os.environ["COHERE_API_KEY"])
                logger.info("Reranker: Cohere")
                return ("cohere", client)
            except Exception as exc:
                logger.warning("Cohere init failed (%s); falling back", exc)

        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("BAAI/bge-reranker-base")
            logger.info("Reranker: BGE cross-encoder")
            return ("cross_encoder", model)
        except ImportError:
            logger.warning("No reranker available")
            return None

    def _rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        if not docs:
            return docs

        kind, model = self.reranker

        try:
            if kind == "cohere":
                res = model.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=[d.page_content for d in docs],
                    top_n=len(docs),
                )
                return [docs[r.index] for r in res.results[:k]]

            scores = model.predict([(query, d.page_content) for d in docs])
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked[:k]]

        except Exception:
            logger.exception("Rerank failed; returning fused order")
            return docs[:k]

    def _check_ready(self) -> None:
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Pehle `create_vectorstore(documents)` call karo.")