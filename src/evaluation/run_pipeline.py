import time
import statistics
from datetime import datetime
from langsmith import Client
from langsmith.evaluation import evaluate
from src.evaluation.evaluators import RAGEvaluators
from src.evaluation.target import predict
from src.config.config import Config
from datetime import datetime, timezone  # ← top pe add karein


def run_evaluation(
    dataset_name: str = "rag-eval-dataset-v1",
    experiment_prefix: str = "rag-doc-intel",
    model_version: str = "1.0.0",
):
    evaluators = RAGEvaluators()

    experiment_metadata = {
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": "gpt-4o-mini",
    }

    print(f"Starting evaluation on dataset: {dataset_name}")
    
    results = evaluate(
        predict,
        data=dataset_name,
        evaluators=[
            evaluators.correctness,
            evaluators.answer_relevancy,
            evaluators.retrieval_relevancy,
            evaluators.groundedness,
        ],
        experiment_prefix=f"{experiment_prefix}-{model_version}",
        metadata=experiment_metadata,
        max_concurrency=2,  # Rate limits se bachne ke liye
    )
    return results


def assert_quality_gates(experiment_name: str, thresholds: dict):
    """
    CI/CD ke liye: agar metrics threshold se neeche hain to fail karein.
    """
    print(f"\nRunning quality gates on: {experiment_name}")
    time.sleep(3)  # LangSmith aggregation ke liye wait

    client = Client(
        api_key=Config.LANGSMITH_API_KEY,
        api_url=Config.LANGSMITH_ENDPOINT
    )
    
    runs = list(client.list_runs(project_name=experiment_name, is_root=True))

    buckets = {k: [] for k in thresholds.keys()}
    for run in runs:
        for fb in client.list_feedback(run_ids=[run.id]):
            if fb.key in buckets and fb.score is not None:
                buckets[fb.key].append(fb.score)

    passed = True
    for metric, scores in buckets.items():
        if not scores:
            continue
        avg = statistics.mean(scores)
        threshold = thresholds.get(metric, 0.75)
        status = "PASS" if avg >= threshold else "FAIL"
        if status == "FAIL":
            passed = False
        print(f"  {metric}: {avg:.3f} (threshold {threshold}) [{status}]")

    if not passed:
        raise SystemExit("\n[ERROR] Quality gates FAILED. Deployment blocked.")
    print("[SUCCESS] All quality gates passed. Safe to deploy.")

if __name__ == "__main__":
    results = run_evaluation(
        dataset_name="rag-eval-dataset-v1",
        experiment_prefix="rag-doc-intel",
        model_version="1.0.0",
    )
    
    # ═══════════════════════════════════════════════════════════════
    # NEW: Print results locally
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for r in results:
        print(f"\nQuestion: {r['example'].inputs['question']}")
        for eval_result in r['evaluation_results']['results']:
            key = eval_result.key
            score = eval_result.score
            comment = eval_result.comment
            print(f"  • {key}: {score:.2f} — {comment[:80]}...")
    
    # Quality gates
    THRESHOLDS = {
        "correctness": 0.80,
        "answer_relevancy": 0.85,
        "retrieval_relevancy": 0.80,
        "groundedness": 0.90,
    }
    
    # Experiment name from results
    exp_name = results.experiment_name if hasattr(results, 'experiment_name') else "rag-doc-intel-1.0.0"
    assert_quality_gates(exp_name, THRESHOLDS)