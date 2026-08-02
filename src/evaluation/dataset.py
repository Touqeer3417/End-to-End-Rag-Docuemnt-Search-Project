import json
from pathlib import Path
from langsmith import Client
from src.config.config import Config  # ← FIXED


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_dataset.json"

def get_local_examples() -> list:
    """Best practice: keep a local JSON backup that is git-tracked."""
    if not DATASET_PATH.exists():
        examples =[
            
        {
            "inputs": {
            "question": "What is the hidden size (d_model) of the Transformer base model?"
            },
            "outputs": {
            "answer": "The hidden size (d_model) of the Transformer base model is 512.",
            "ground_truth": "512"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "easy",
            "type": "direct_fact"
            }
        },
        {
            "inputs": {
            "question": "How many attention heads are used in the Transformer base model and what is the dimension of each head?"
            },
            "outputs": {
            "answer": "The Transformer base model uses 8 attention heads, and each head has a dimension of 64 (512 / 8 = 64).",
            "ground_truth": "8 heads, 64 dimensions each"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "medium",
            "type": "direct_fact"
            }
        },
        {
            "inputs": {
            "question": "What is the formula for positional encoding used in the Transformer?"
            },
            "outputs": {
            "answer": "The positional encoding uses sine and cosine functions: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)).",
            "ground_truth": "Sine and cosine functions with wavelength geometric progression from 2π to 10000·2π"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "hard",
            "type": "formula"
            }
        },
        {
            "inputs": {
            "question": "What optimizer was used to train the Transformer and what were its key hyperparameters?"
            },
            "outputs": {
            "answer": "The Adam optimizer was used with β1 = 0.9, β2 = 0.98, and ε = 10^-9. The learning rate varied according to a custom schedule.",
            "ground_truth": "Adam optimizer with β1=0.9, β2=0.98, ε=10^-9"
            },
            "metadata": {
            "category": "training",
            "difficulty": "medium",
            "type": "hyperparameter"
            }
        },
        {
            "inputs": {
            "question": "What was the BLEU score achieved by the Transformer big model on the English-to-German newstest2014 dataset?"
            },
            "outputs": {
            "answer": "The Transformer big model achieved a BLEU score of 28.4 on the English-to-German newstest2014 dataset, establishing a new state-of-the-art.",
            "ground_truth": "28.4 BLEU"
            },
            "metadata": {
            "category": "results",
            "difficulty": "easy",
            "type": "metric"
            }
        },
        {
            "inputs": {
            "question": "Why does the Transformer use scaled dot-product attention instead of standard dot-product attention?"
            },
            "outputs": {
            "answer": "For large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. To counteract this effect, the dot products are scaled by 1/√dk.",
            "ground_truth": "To prevent large dot product magnitudes from pushing softmax into regions with small gradients"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "medium",
            "type": "reasoning"
            }
        },
        {
            "inputs": {
            "question": "What is the time complexity of self-attention compared to recurrent layers, and why is this advantageous?"
            },
            "outputs": {
            "answer": "Self-attention has O(1) sequential operations and O(n²) complexity per layer, while recurrent layers have O(n) sequential operations. Self-attention is faster when sequence length n is smaller than representation dimensionality d.",
            "ground_truth": "Self-attention: O(1) sequential ops, O(n²) per layer; RNN: O(n) sequential ops. Advantageous for shorter sequences"
            },
            "metadata": {
            "category": "complexity",
            "difficulty": "hard",
            "type": "comparison"
            }
        },
        {
            "inputs": {
            "question": "What is the purpose of masked self-attention in the decoder, and where is it applied?"
            },
            "outputs": {
            "answer": "Masked self-attention prevents positions from attending to subsequent positions in the decoder. This ensures that predictions for position i can depend only on known outputs at positions less than i, preserving the auto-regressive property.",
            "ground_truth": "Prevents attending to future positions, preserving auto-regressive property in decoder"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "medium",
            "type": "concept"
            }
        },
        {
            "inputs": {
            "question": "What is the feed-forward network dimension (d_ff) in the Transformer base model?"
            },
            "outputs": {
            "answer": "The feed-forward network dimension (d_ff) in the Transformer base model is 2048.",
            "ground_truth": "2048"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "easy",
            "type": "direct_fact"
            }
        },
        {
            "inputs": {
            "question": "How does the Transformer handle the order of tokens since it has no recurrence or convolution?"
            },
            "outputs": {
            "answer": "The Transformer handles token order by adding positional encodings to the input embeddings at the bottom of the encoder and decoder stacks. These encodings have the same dimension as the embeddings and use sine and cosine functions of different frequencies.",
            "ground_truth": "Positional encodings added to input embeddings using sine/cosine functions"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "easy",
            "type": "concept"
            }
        },
        {
            "inputs": {
            "question": "What was the training setup for the base model in terms of hardware and training time?"
            },
            "outputs": {
            "answer": "The base model was trained on 8 P100 GPUs for 12 hours, completing 100,000 steps with approximately 25,000 total training tokens per batch.",
            "ground_truth": "8 P100 GPUs, 12 hours, 100,000 steps"
            },
            "metadata": {
            "category": "training",
            "difficulty": "medium",
            "type": "setup"
            }
        },
        {
            "inputs": {
            "question": "What is label smoothing and what value was used in the Transformer training?"
            },
            "outputs": {
            "answer": "Label smoothing is a regularization technique that prevents the model from becoming too confident. The Transformer used a label smoothing value of ε_ls = 0.1, which hurt perplexity but improved BLEU score and accuracy.",
            "ground_truth": "ε_ls = 0.1, improves accuracy and BLEU despite hurting perplexity"
            },
            "metadata": {
            "category": "training",
            "difficulty": "medium",
            "type": "technique"
            }
        },
        {
            "inputs": {
            "question": "What is the learning rate schedule formula used during Transformer training?"
            },
            "outputs": {
            "answer": "The learning rate increases linearly for the first warmup_steps (4000) and then decreases proportionally to the inverse square root of the step number: lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5)).",
            "ground_truth": "lrate = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))"
            },
            "metadata": {
            "category": "training",
            "difficulty": "hard",
            "type": "formula"
            }
        },
        {
            "inputs": {
            "question": "How many layers are in the Transformer encoder and decoder respectively?"
            },
            "outputs": {
            "answer": "Both the encoder and decoder in the Transformer base model consist of a stack of N = 6 identical layers.",
            "ground_truth": "6 layers each"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "easy",
            "type": "direct_fact"
            }
        },
        {
            "inputs": {
            "question": "What is the dropout rate applied in the Transformer model?"
            },
            "outputs": {
            "answer": "A dropout rate of 0.1 was applied to the output of each sub-layer, before it is added to the sub-layer input and normalized. Dropout was also applied to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.",
            "ground_truth": "0.1 dropout on sub-layer outputs, embeddings, and positional encodings"
            },
            "metadata": {
            "category": "training",
            "difficulty": "easy",
            "type": "hyperparameter"
            }
        },
        {
            "inputs": {
            "question": "What is the difference between the Transformer base and big models in terms of hyperparameters?"
            },
            "outputs": {
            "answer": "The big model uses N=6 layers, d_model=1024, d_ff=4096, and h=16 attention heads, while the base model uses d_model=512, d_ff=2048, and h=8. The big model also uses a larger batch size (248K vs 25K tokens) and dropout rate of 0.3.",
            "ground_truth": "Big: d_model=1024, d_ff=4096, h=16, dropout=0.3; Base: d_model=512, d_ff=2048, h=8, dropout=0.1"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "hard",
            "type": "comparison"
            }
        },
        {
            "inputs": {
            "question": "What does the paper claim about the first transduction model based entirely on attention?"
            },
            "outputs": {
            "answer": "The paper claims that the Transformer is the first transduction model based entirely on attention, replacing recurrence with attention mechanisms to draw global dependencies between input and output.",
            "ground_truth": "First transduction model based entirely on attention, replacing recurrence"
            },
            "metadata": {
            "category": "claims",
            "difficulty": "easy",
            "type": "quote"
            }
        },
        {
            "inputs": {
            "question": "What is multi-head attention and why is it beneficial?"
            },
            "outputs": {
            "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. It uses h=8 parallel attention layers with projected queries, keys, and values.",
            "ground_truth": "Allows attending to different representation subspaces in parallel, h=8 heads"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "medium",
            "type": "concept"
            }
        },
        {
            "inputs": {
            "question": "What is the maximum sequence length used during training and what positional encoding method was compared but not used in the final model?"
            },
            "outputs": {
            "answer": "The maximum sequence length was 100 tokens. Learned positional embeddings were compared to sinusoidal encodings but produced nearly identical results, so the sinusoidal version was chosen because it might allow the model to extrapolate to longer sequences.",
            "ground_truth": "Max length 100; learned embeddings tested but sinusoidal chosen for extrapolation"
            },
            "metadata": {
            "category": "architecture",
            "difficulty": "hard",
            "type": "multi_hop"
            }
        },
        {
            "inputs": {
            "question": "What was the BLEU score on English-to-French newstest2014 and how long did it take to train?"
            },
            "outputs": {
            "answer": "The Transformer big model achieved 41.0 BLEU on English-to-French newstest2014, and the base model took 3.5 days on 8 P100 GPUs to train.",
            "ground_truth": "41.0 BLEU; base model trained in 3.5 days on 8 P100 GPUs"
            },
            "metadata": {
            "category": "results",
            "difficulty": "medium",
            "type": "metric"
            }
        }
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