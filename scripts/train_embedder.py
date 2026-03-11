"""
train_embedder.py – Fine-tune the embedding model on plant data pairs.

Offline script:
    python -m scripts.train_embedder

Uses settings for default model path.
"""

from pathlib import Path

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

from src.config.settings import EMBEDDING_MODEL_DIR

DATA_PATH = Path("data/processed/train_pairs.jsonl")
BASE_MODEL = EMBEDDING_MODEL_DIR
OUT_DIR = Path("models/garssa-embed-v1")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH} – run scripts.build_training_data first."
        )

    ds = load_dataset("json", data_files=str(DATA_PATH), split="train")
    print(f"Training examples: {len(ds)}")

    examples = [
        InputExample(texts=[row["query"], row["positive"]])
        for row in ds
    ]

    # Hold-out for evaluation
    test_ratio = 0.15
    split_idx = int(len(examples) * (1 - test_ratio))
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]

    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = evaluation.InformationRetrievalEvaluator(
        queries={str(i): ex.texts[0] for i, ex in enumerate(test_examples)},
        corpus={str(i): ex.texts[1] for i, ex in enumerate(test_examples)},
        relevant_docs={str(i): {str(i)} for i in range(len(test_examples))},
        name="garssa-ir-eval",
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=6,
        output_path=str(OUT_DIR),
        show_progress_bar=True,
    )

    print(f"✅ Model saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
