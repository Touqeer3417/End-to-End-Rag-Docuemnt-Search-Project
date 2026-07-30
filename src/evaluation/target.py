from pathlib import Path
from langchain_core.runnables import RunnableConfig

# Aapke existing imports
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorStore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
os.environ["USER_AGENT"] = "Touqeer-RAG-Eval/1.0"

_pipeline = None

def _init_pipeline():
    """Initialize RAG pipeline for evaluation (no Streamlit cache)."""
    print("🔧 Initializing RAG pipeline for evaluation...")
    
    llm = Config.get_llm()
    print("✅ LLM loaded")
    
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()
    print("✅ VectorStore initialized")
    
    # Try to load existing vectorstore first (fast)
    try:
        retriever = vector_store.get_retriever()
        print("✅ Existing vectorstore loaded")
    except Exception as e:
        print(f"⚠️  No existing vectorstore found ({e}), rebuilding...")
        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_urls(urls)
        print(f"✅ Processed {len(documents)} URL documents")
        
        pdf_path = Path("data/attention.pdf")
        if pdf_path.exists():
            pdf_loader = PyPDFLoader(str(pdf_path))
            pdf_docs = pdf_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            pdf_chunks = text_splitter.split_documents(pdf_docs)
            documents.extend(pdf_chunks)
            print(f"✅ Added {len(pdf_chunks)} PDF chunks")
        
        vector_store.create_vectorstore(documents)
        retriever = vector_store.get_retriever()
        print(f"✅ Vectorstore created with {len(documents)} total chunks")
    
    graph_builder = GraphBuilder(
        retriever=retriever,
        llm=llm
    )
    graph_builder.build()
    print("✅ Graph built successfully")
    return graph_builder

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _init_pipeline()
    return _pipeline


def predict(inputs: dict) -> dict:
    """
    Target function for LangSmith evaluate().
    Must return: answer, context
    """
    question = inputs["question"]
    graph = get_pipeline()
    
    # Invoke graph (non-streaming)
    result = graph.invoke(
        question,
        config=RunnableConfig(metadata={"eval": True, "source": "batch_evaluation"})
    )
    
    # Extract answer (handle both string and AIMessage)
    answer = result.get("answer", "")
    if hasattr(answer, "content"):
        answer = answer.content
    elif isinstance(answer, list) and len(answer) > 0 and hasattr(answer[-1], "content"):
        answer = answer[-1].content
    
    # Extract context (handle Document objects or strings)
    raw_context = result.get("context", result.get("documents", []))
    normalized_context = []
    for doc in raw_context if isinstance(raw_context, list) else [raw_context]:
        if hasattr(doc, "page_content"):
            normalized_context.append(doc.page_content)
        elif isinstance(doc, str):
            normalized_context.append(doc)
    
    return {
        "answer": str(answer),
        "context": normalized_context,
    }