import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants
MODEL_NAME = "DeepPavlov/rubert-base-cased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer (loaded once)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)


def get_embedding(text: str) -> torch.Tensor:
    """Compute mean pooling embedding for a single text."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over the sequence dimension
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding


def cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    sim = torch.nn.functional.cosine_similarity(emb1, emb2)
    return sim.item()


def compute_skill_stats(skills: List[str], position: str) -> Dict[str, float]:
    """Compute statistics for skill similarities."""
    if not skills:
        return {
            "mean_similarity": 0.0,
            "count_above_05": 0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "std_similarity": 0.0,
        }

    similarities = [cosine_similarity(skill, position) for skill in skills]
    return {
        "mean_similarity": np.mean(similarities),
        "count_above_05": sum(1 for s in similarities if s > 0.5),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "std_similarity": np.std(similarities),
    }


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Process dataset to add skill similarity statistics."""
    def process_row(row: pd.Series) -> pd.Series:
        skills = [s.strip() for s in str(row["key_skills"]).split(",") if s.strip()]
        stats = compute_skill_stats(skills, row["position"])
        return pd.Series(stats)

    new_cols = df.apply(process_row, axis=1)
    return pd.concat([df, new_cols], axis=1)


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("../data/client_dataset_csv.csv", index_col=0).head(3)
    print(f"Dataset shape: {data.shape}")

    # Process and display results
    processed_data = process_dataset(data)
    print(
        processed_data[
            [
                "mean_similarity",
                "count_above_05",
                "min_similarity",
                "max_similarity",
                "std_similarity",
            ]
        ]
    )
