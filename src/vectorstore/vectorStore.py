"""Vector store module for document embedding and retrieval using Qdrant."""

from typing import List, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class VectorStore:
    VECTOR_SIZE: int = 1536
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "documents"
    ):
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            check_compatibility=False,
            timeout=60
        )
        self.vectorstore: Optional[QdrantVectorStore] = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document], force: bool = False) -> None:
        if not documents:
            raise ValueError("Documents list khali hai!")
        
        exists = self.collection_name in [c.name for c in self.client.get_collections().collections]
        
        if exists and not force:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding
            )
        elif exists and force:
            self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE)
            )
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding
            )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE)
            )
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding
            )
        
        self.vectorstore.add_documents(documents)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.vectorstore.similarity_search(query=query, k=k)