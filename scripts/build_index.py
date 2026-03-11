"""
build_index.py – Build FAISS index from processed cards.

Offline script: run once after generating cards.jsonl.
    python -m scripts.build_index

Uses settings.py paths so there's a single source of truth.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import EMBEDDING_MODEL_DIR, INDEX_PATH, META_PATH

CARDS_PATH = Path("data/processed/cards.jsonl")
MODEL_DIR = Path(EMBEDDING_MODEL_DIR)

OUT_DIR = Path(INDEX_PATH).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

_INDEX_PATH = Path(INDEX_PATH)
_META_PATH = Path(META_PATH)


def load_cards():
    cards = []
    with CARDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if text:
                cards.append(text)
    return cards


def main():
    if not CARDS_PATH.exists():
        raise FileNotFoundError(f"Missing {CARDS_PATH}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing fine-tuned model at {MODEL_DIR}")

    cards = load_cards()
    print(f"Loaded cards: {len(cards)}")

    model = SentenceTransformer(str(MODEL_DIR))

    emb = model.encode(cards, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine (because normalized) via inner-product
    index.add(emb)

    faiss.write_index(index, str(_INDEX_PATH))
    _META_PATH.write_text(
        json.dumps({"cards": cards}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ Saved index: {_INDEX_PATH}")
    print(f"✅ Saved meta : {_META_PATH}")


if __name__ == "__main__":
    main()
