"""Graph builder for LangGraph workflow (Corrective RAG)."""

import logging
import threading
from functools import wraps
from typing import Callable, Optional  # ← NEW: Optional import

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.nodes import RAGNodes

logger = logging.getLogger(__name__)

MAX_QUERY_TRANSFORMS = 2   # loop guard: cap retries before forcing generate
NODE_TIMEOUT_SECONDS = 30  # per-node timeout


def with_error_handling(node_name: str):
    """Wrap a node fn so exceptions don't crash whole graph.
    On failure, logs and returns partial state update marking error."""
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(state: RAGState, *args, **kwargs):
            try:
                return fn(state, *args, **kwargs)
            except Exception as e:
                logger.exception("Node '%s' failed: %s", node_name, e)
                return {"error": f"{node_name}_failed: {e}"}
        return wrapper
    return decorator


def route_after_retrieval(state: RAGState) -> str:
    """Empty docs or prior error → skip grading."""
    if getattr(state, "error", None):
        return "transform_query"
    if not state.retrieved_docs:
        return "transform_query"
    return "grade_documents"


def decide_to_generate(state: RAGState) -> str:
    """Loop guard: cap query transforms to avoid infinite retry loop."""
    transform_count = getattr(state, "transform_count", 0)
    if state.web_search_needed and transform_count < MAX_QUERY_TRANSFORMS:
        return "transform_query"
    return "generate"


class GraphBuilder:
    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
        self._lock = threading.Lock()  # thread-safe lazy build

    def build(self):
        if self.graph is not None:
            return self.graph

        with self._lock:
            if self.graph is not None:  # double-check inside lock
                return self.graph

            builder = StateGraph(RAGState)

            builder.add_node("retriever", with_error_handling("retriever")(self.nodes.retrieve_docs))
            builder.add_node("grade_documents", with_error_handling("grade_documents")(self.nodes.grade_documents))
            builder.add_node("transform_query", with_error_handling("transform_query")(self.nodes.transform_query))
            builder.add_node("web_search", with_error_handling("web_search")(self.nodes.web_search))
            builder.add_node("responder", with_error_handling("responder")(self.nodes.generate_answer))

            builder.set_entry_point("retriever")

            builder.add_conditional_edges(
                "retriever",
                route_after_retrieval,
                {
                    "grade_documents": "grade_documents",
                    "transform_query": "transform_query",
                },
            )

            builder.add_conditional_edges(
                "grade_documents",
                decide_to_generate,
                {
                    "transform_query": "transform_query",
                    "generate": "responder",
                },
            )

            builder.add_edge("transform_query", "web_search")
            builder.add_edge("web_search", "responder")
            builder.add_edge("responder", END)

            self.graph = builder.compile()
            return self.graph

    # ═══════════════════════════════════════════════════════════════
    # NEW: invoke() — for evaluation & API calls (non-streaming)
    # Returns normalized dict: {answer, context, sources}
    # ═══════════════════════════════════════════════════════════════
    def invoke(self, question: str, config: Optional[dict] = None) -> dict:
        if not question or not question.strip():
            raise ValueError("question empty")
        
        graph = self.build()
        initial_state = RAGState(question=question)
        
        # Merge defaults with user config (for LangSmith metadata/tracing)
        default_config = {
            "recursion_limit": 25,
            "configurable": {"timeout": NODE_TIMEOUT_SECONDS},
        }
        if config:
            merged = {**default_config, **config}
            if "configurable" in default_config and "configurable" in config:
                merged["configurable"] = {**default_config["configurable"], **config["configurable"]}
        else:
            merged = default_config
        
        final_state = graph.invoke(initial_state, config=merged)
        return self._normalize_state(final_state)

    # ═══════════════════════════════════════════════════════════════
    # NEW: Helper to convert RAGState → plain dict for evaluators
    # ═══════════════════════════════════════════════════════════════
    def _normalize_state(self, state) -> dict:
        """Extract answer, context, and sources from final state."""
        # Handle both dict (TypedDict) and object (Pydantic/dataclass)
        if isinstance(state, dict):
            answer = state.get("answer", "")
            docs = state.get("retrieved_docs", [])
            sources = state.get("sources", [])
        else:
            answer = getattr(state, "answer", "")
            docs = getattr(state, "retrieved_docs", [])
            sources = getattr(state, "sources", [])
        
        # Normalize documents to string list
        context = []
        for doc in docs if docs else []:
            if hasattr(doc, "page_content"):
                context.append(str(doc.page_content))
            elif isinstance(doc, dict):
                context.append(str(doc.get("page_content", doc.get("content", str(doc)))))
            else:
                context.append(str(doc))
        
        # Normalize answer (handle AIMessage / string / list)
        if hasattr(answer, "content"):
            answer = answer.content
        elif isinstance(answer, list) and len(answer) > 0 and hasattr(answer[-1], "content"):
            answer = answer[-1].content
        
        return {
            "answer": str(answer) if answer else "",
            "context": context,
            "sources": sources if sources else [],
        }

    def run(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("question empty")
        graph = self.build()
        initial_state = RAGState(question=question)
        try:
            return graph.invoke(
                initial_state,
                config={"recursion_limit": 25, "configurable": {"timeout": NODE_TIMEOUT_SECONDS}},
            )
        except Exception as e:
            logger.exception("Graph run failed for question=%r: %s", question, e)
            raise

    # ═══════════════════════════════════════════════════════════════
    # MODIFIED: stream() now accepts config for LangSmith tracing
    # ═══════════════════════════════════════════════════════════════
    def stream(self, question: str, config: Optional[dict] = None):
        if not question or not question.strip():
            raise ValueError("question empty")
        graph = self.build()
        initial_state = RAGState(question=question)
        
        # Merge default recursion limit with user config
        merged_config = {"recursion_limit": 25}
        if config:
            merged_config = {**merged_config, **config}
            if "configurable" in config:
                base_configurable = merged_config.get("configurable", {})
                merged_config["configurable"] = {**base_configurable, **config["configurable"]}
        
        try:
            for chunk, metadata in graph.stream(
                initial_state,
                stream_mode="messages",
                config=merged_config,
            ):
                yield chunk, metadata
        except Exception as e:
            logger.exception("Graph stream failed for question=%r: %s", question, e)
            raise