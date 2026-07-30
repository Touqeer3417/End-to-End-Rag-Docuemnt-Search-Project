import json
from pathlib import Path
from langsmith import Client
from src.config.config import Config  # ← FIXED


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_dataset.json"

def get_local_examples() -> list:
    """Best practice: keep a local JSON backup that is git-tracked."""
    if not DATASET_PATH.exists():
        examples = [
            {
                "inputs": {"question": "What is the return policy timeframe?"},
                "outputs": {
                    "answer": "Customers may return items within 30 days of purchase.",
                    "ground_truth": "30-day return window"
                },
                "metadata": {"category": "policy", "difficulty": "easy"}
            },
            {
                "inputs": {"question": "How do I upgrade my subscription and will I be refunded for the current month?"},
                "outputs": {
                    "answer": "You can upgrade anytime; the difference is prorated and the current month is not refunded but credited.",
                    "ground_truth": "Prorated upgrade, no refund but credit applied"
                },
                "metadata": {"category": "billing", "difficulty": "hard"}
            },
            {
                "inputs": {"question": "What security certifications does the platform hold?"},
                "outputs": {
                    "answer": "The platform is SOC 2 Type II and ISO 27001 certified.",
                    "ground_truth": "SOC 2 Type II, ISO 27001"
                },
                "metadata": {"category": "security", "difficulty": "medium"}
            },
        ]
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        return examples

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def sync_dataset_to_langsmith(dataset_name: str = "rag-eval-dataset-v1"):
    client = Client(
        api_key=Config.LANGSMITH_API_KEY,
        api_url=Config.LANGSMITH_ENDPOINT)
    
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists. Skipping creation.")
        return dataset
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Production RAG evaluation dataset with ground truth",
    )

    examples = get_local_examples()
    for ex in examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            metadata=ex.get("metadata", {}),
            dataset_id=dataset.id,
        )
    print(f"Uploaded {len(examples)} examples to dataset '{dataset_name}'.")
    return dataset

if __name__ == "__main__":
    sync_dataset_to_langsmith()