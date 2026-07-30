"""LangGraph nodes for RAG workflow (Corrective RAG)."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from src.state.rag_state import RAGState
from src.config.config import Config

logger = logging.getLogger(__name__)

GRADE_WORKERS = 5          # parallel grading threads
WEB_SEARCH_MAX_RESULTS = 3


class GradeDocuments(BaseModel):
    """Binary relevance grade for a retrieved document."""

    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


class RAGNodes:
    """Contains node functions for Corrective RAG workflow."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.grader = llm.with_structured_output(GradeDocuments)
        self.websearch_tool = TavilySearch(
            max_results=WEB_SEARCH_MAX_RESULTS, tavily_api_key=Config.TAVILY_API_KEY
        )

    def retrieve_docs(self, state: RAGState) -> dict:
        """Retrieve relevant documents node."""
        try:
            docs = self.retriever.invoke(state.question)
            return {"retrieved_docs": docs}
        except Exception as e:
            logger.exception("retrieve_docs failed: %s", e)
            return {"retrieved_docs": [], "error": f"retrieve_failed: {e}"}


    def _grade_one(self, doc: Document, question: str) -> tuple[Document, bool]:
        """Grade a single doc. Returns (doc, is_relevant)."""
        prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
            f"Document:\n{doc.page_content}\n\n"
            f"Question: {question}\n\n"
            "Give a binary score 'yes' or 'no'. 'yes' means the document is relevant to the question."
        )
        try:
            result = self.grader.invoke(prompt)
            return doc, result.binary_score.strip().lower() == "yes"
        except Exception as e:
            logger.exception("grade_one failed: %s", e)
            return doc, False  # fail-safe: treat as irrelevant, triggers web search


    def grade_documents(self, state: RAGState) -> dict:
        """
        Grade each retrieved doc as relevant/irrelevant, in parallel.
        Keeps only relevant docs. Flags web_search_needed if any doc fails.
        """
        if not state.retrieved_docs:
            return {"retrieved_docs": [], "web_search_needed": True}

        relevant_docs: List[Document] = []
        web_search_needed = False

        with ThreadPoolExecutor(max_workers=GRADE_WORKERS) as pool:
            futures = [
                pool.submit(self._grade_one, doc, state.question)
                for doc in state.retrieved_docs
            ]
            for future in as_completed(futures):
                doc, is_relevant = future.result()
                if is_relevant:
                    relevant_docs.append(doc)
                else:
                    web_search_needed = True

        return {"retrieved_docs": relevant_docs, "web_search_needed": web_search_needed}


    def transform_query(self, state: RAGState) -> dict:
        """Rewrite question into a clearer, web-search-optimized query. Tracks retry count for loop guard."""
        transform_count = getattr(state, "transform_count", 0) + 1
        try:
            prompt = (
                "Rewrite the question below into a clearer, keyword-focused query optimized for web search.\n"
                "Keep the original meaning intact. Return only the rewritten query, nothing else.\n\n"
                f"Original question: {state.question}"
            )
            response = self.llm.invoke(prompt)
            return {
                "rewritten_query": response.content.strip(),
                "transform_count": transform_count,
            }
        except Exception as e:
            logger.exception("transform_query failed: %s", e)
            return {"rewritten_query": state.question, "transform_count": transform_count}


    def web_search(self, state: RAGState) -> dict:
        """Fallback web search node, runs when local docs graded irrelevant."""
        query = state.rewritten_query or state.question
        try:
            response = self.websearch_tool.invoke({"query": query})
            web_docs = [
                Document(
                    page_content=r.get("content", ""),
                    metadata={"source": r.get("url", "web_search")},
                )
                for r in response.get("results", [])
            ]
            return {"retrieved_docs": state.retrieved_docs + web_docs}
        except Exception as e:
            logger.exception("web_search failed: %s", e)
            return {"error": f"web_search_failed: {e}"}


    def generate_answer(self, state: RAGState) -> dict:
        """Generate answer from (corrected) retrieved documents node."""
        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)
        prompt = (
            f"Answer the question based on the context.\n\nContext:\n{context}\n\n"
            f"Question: {state.question}"
        )
        try:
            full_text = "".join(
                chunk.content for chunk in self.llm.stream(prompt)
            )
            return {"answer": full_text}
        except Exception as e:
            logger.exception("generate_answer failed: %s", e)
            return {"answer": "", "error": f"generate_failed: {e}"}