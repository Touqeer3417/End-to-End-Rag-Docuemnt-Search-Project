"""Graph builder for LangGraph workflow (Corrective RAG)"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.nodes import RAGNodes


def decide_to_generate(state: RAGState) -> str:
    """Route: straight to generate, or transform_query -> web_search first"""
    if state.web_search_needed:
        return "transform_query"
    return "generate"


class GraphBuilder:
    """Builds and manages the Corrective RAG LangGraph workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize graph builder

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        """
        Build the Corrective RAG workflow graph

        Returns:
            Compiled graph instance
        """
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("grade_documents", self.nodes.grade_documents)
        builder.add_node("transform_query", self.nodes.transform_query)
        builder.add_node("web_search", self.nodes.web_search)
        builder.add_node("responder", self.nodes.generate_answer)

        # Entry point
        builder.set_entry_point("retriever")

        # Always grade after retrieving
        builder.add_edge("retriever", "grade_documents")

        # CRAG branch point: generate directly, or correct first
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

        # Compile graph
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str) -> dict:
        """
        Run the Corrective RAG workflow

        Args:
            question: User question

        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
    
    def stream(self, question: str):
        """
        Stream LLM tokens as they generate

        Args:
            question: User question

        Yields:
            tuple: (message_chunk, metadata)
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        for item in self.graph.stream(initial_state, stream_mode="messages"):
            if isinstance(item, tuple) and len(item) == 2:
                yield item
            else:
                yield item, {}
        