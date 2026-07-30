from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from src.config.config import Config


class Grade(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Score between 0.0 and 1.0")
    reasoning: str = Field(..., description="One-sentence justification")


class RAGEvaluators:
    """
    Production-grade LLM-as-a-Judge evaluators.
    """

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        self.judge = ChatOpenAI(
            model=judge_model,
            temperature=0,
            api_key=Config.OPENAI_API_KEY if hasattr(Config, 'OPENAI_API_KEY') else Config.GROQ_API_KEY,
        ).with_structured_output(Grade)

    def _grade(self, system: str, human_template: str, **kwargs) -> Grade:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", human_template),
        ])
        chain = prompt | self.judge
        return chain.invoke(kwargs)

    # ═══════════════════════════════════════════════════════════════
    # 1. CORRECTNESS
    # Purpose: Is the answer factually equivalent to ground truth?
    # Score: 1.0 = fully correct, 0.0 = wrong, decimals = partial
    # ═══════════════════════════════════════════════════════════════
    def correctness(self, run: Run, example: Example) -> EvaluationResult:
        predicted = run.outputs.get("answer", "")
        expected = example.outputs.get("answer", "") if example.outputs else ""
        
        if not expected:
            return EvaluationResult(key="correctness", score=None, comment="No ground truth")

        grade = self._grade(
            system=(
                "You are a strict factual grader. Compare Predicted Answer to Expected Answer. "
                "Score 1.0 if fully correct and complete. Score 0.0 if wrong or contradictory. "
                "Use decimals for partial correctness or missing details."
            ),
            human_template="Expected Answer:\n{expected}\n\nPredicted Answer:\n{predicted}",
            expected=expected,
            predicted=predicted,
        )
        return EvaluationResult(key="correctness", score=grade.score, comment=grade.reasoning)

    # ═══════════════════════════════════════════════════════════════
    # 2. ANSWER RELEVANCY
    # Purpose: Does the answer directly address the user's question?
    # Score: 1.0 = directly relevant; 0.0 = off-topic
    # Note: Ignore factual correctness here. Only measure topical alignment.
    # ═══════════════════════════════════════════════════════════════
    def answer_relevancy(self, run: Run, example: Example) -> EvaluationResult:
        question = example.inputs.get("question", "")
        predicted = run.outputs.get("answer", "")

        grade = self._grade(
            system=(
                "Evaluate whether the answer is relevant to the question. "
                "Score 1.0 if the answer directly addresses the question. "
                "Score 0.0 if off-topic, evasive, or unrelated. "
                "Do NOT penalize factual errors—only relevance."
            ),
            human_template="Question:\n{question}\n\nAnswer:\n{predicted}",
            question=question,
            predicted=predicted,
        )
        return EvaluationResult(key="answer_relevancy", score=grade.score, comment=grade.reasoning)

    # ═══════════════════════════════════════════════════════════════
    # 3. RETRIEVAL RELEVANCY
    # Purpose: Are retrieved documents relevant to the question?
    # Score: 1.0 = all context useful; 0.0 = completely irrelevant
    # ═══════════════════════════════════════════════════════════════
    def retrieval_relevancy(self, run: Run, example: Example) -> EvaluationResult:
        question = example.inputs.get("question", "")
        contexts = run.outputs.get("context", [])
        context_str = "\n\n---\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)

        grade = self._grade(
            system=(
                "Evaluate retrieval quality. Given the question and retrieved context, "
                "score 1.0 if context is highly relevant and sufficient. "
                "Score 0.0 if irrelevant or noisy. Penalize partial irrelevance."
            ),
            human_template="Question:\n{question}\n\nRetrieved Context:\n{context}",
            question=question,
            context=context_str,
        )
        return EvaluationResult(key="retrieval_relevancy", score=grade.score, comment=grade.reasoning)

    # ═══════════════════════════════════════════════════════════════
    # 4. GROUNDEDNESS (FAITHFULNESS)
    # Purpose: Is every claim supported by retrieved context?
    # Score: 1.0 = fully grounded; 0.0 = hallucinations present
    # Best Practice: Flag any extrapolation or external knowledge.
    # ═══════════════════════════════════════════════════════════════
    def groundedness(self, run: Run, example: Example) -> EvaluationResult:
        predicted = run.outputs.get("answer", "")
        contexts = run.outputs.get("context", [])
        context_str = "\n\n---\n\n".join(contexts) if isinstance(contexts, list) else str(contexts)

        grade = self._grade(
            system=(
                "You are a hallucination detector. Compare the answer to the provided context. "
                "Score 1.0 only if every claim is explicitly supported by the context. "
                "Score 0.0 if the answer contains hallucination, extrapolation, or unsupported facts. "
                "Be strict."
            ),
            human_template="Context:\n{context}\n\nAnswer:\n{predicted}",
            context=context_str,
            predicted=predicted,
        )
        return EvaluationResult(key="groundedness", score=grade.score, comment=grade.reasoning)