"""LangGraph nodes for RAG workflow (Corrective RAG)"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from src.state.rag_state import RAGState
from src.config.config import Config


class GradeDocuments(BaseModel):
    """Binary relevance grade for a retrieved document."""

    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


class RAGNodes:
    """Contains node functions for Corrective RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
        self.grader = llm.with_structured_output(GradeDocuments)
        self.websearch_tool = TavilySearch(max_results=3, tavily_api_key=Config.TAVILY_API_KEY)

    def retrieve_docs(self, state: RAGState) -> dict:
        """
        Retrieve relevant documents node

        Args:
            state: Current RAG state

        Returns:
            Partial state update with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return {"retrieved_docs": docs}

    def grade_documents(self, state: RAGState) -> dict:
        """
        Grade each retrieved doc as relevant/irrelevant.
        Keeps only relevant docs. Flags web_search_needed if any doc fails.

        Args:
            state: Current RAG state with retrieved documents

        Returns:
            Partial state update: filtered docs + web_search_needed flag
        """
        relevant_docs: List[Document] = []
        web_search_needed = False

        for doc in state.retrieved_docs:
            prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.

Document:
{doc.page_content}

Question: {state.question}

Give a binary score 'yes' or 'no'. 'yes' meuans the document is relevant to the question."""
            result = self.grader.invoke(prompt)
            if result.binary_score.strip().lower() == "yes":
                relevant_docs.append(doc)
            else:
                web_search_needed = True

        return {"retrieved_docs": relevant_docs, "web_search_needed": web_search_needed}

    def transform_query(self, state: RAGState) -> dict:
        """
        Rewrite question into a clearer, web-search-optimized query.

        Args:
            state: Current RAG state

        Returns:
            Partial state update with rewritten_query
        """
        prompt = f"""Rewrite the question below into a clearer, keyword-focused query optimized for web search.
Keep the original meaning intact. Return only the rewritten query, nothing else.

Original question: {state.question}"""
        response = self.llm.invoke(prompt)
        return {"rewritten_query": response.content.strip()}

    def web_search(self, state: RAGState) -> dict:
        """
        Fallback web search node, runs when local docs graded irrelevant.

        Args:
            state: Current RAG state

        Returns:
            Partial state update with web results appended to retrieved_docs
        """
        query = state.rewritten_query or state.question
        response = self.websearch_tool.invoke({"query": query})

        web_docs = [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web_search")},
            )
            for r in response.get("results", [])
        ]
        return {"retrieved_docs": state.retrieved_docs + web_docs}

    def generate_answer(self, state: RAGState) -> dict:
        """
        Generate answer from (corrected) retrieved documents node

        Args:
            state: Current RAG state with retrieved documents

        Returns:
            Partial state update with generated answer
        """
        context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {state.question}"""

        full_text = ""
        for chunk in self.llm.stream(prompt):
            full_text += chunk.content

        return {"answer": full_text}
