import os
from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
import time
from custom_benchmark.common import CacheEvalItem, MOCK_ANSWER_PREFIX
import requests
import random
import argparse
from gptcache.adapter.api import (
    get,
    put,
    init_similar_cache,
    init_similar_cache_from_config,
)

# Set seed for reproducibility
random.seed(42)

@dataclass
class PawnRow:
    id: Optional[str] = None
    sentence1: Optional[str] = None
    sentence2: Optional[str] = None
    label: Optional[Union[int, str]] = None

    @property
    def should_be_evacuated(self) -> bool:
        """
        Determine if the row should be caught based on the label.
        """
        return self.label == "1" # Label 1 is same meaning, so should be caught
    
@dataclass
class PawnRowEval:
    id: str
    cached: bool


def choose_samples(pawn_rows: list[PawnRow], sample_size: int = 10, balanced: bool = True) -> list[PawnRow]:
    """
    Choose a sample of rows from the PAWS dataset.
    If balanced is True, ensure an equal number of positive and negative samples.
    """
    if balanced:
        positive_samples = [row for row in pawn_rows if row.should_be_evacuated]
        negative_samples = [row for row in pawn_rows if not row.should_be_evacuated]
        num_positive = min(len(positive_samples), sample_size // 2)
        num_negative = sample_size - num_positive
        return random.sample(positive_samples, num_positive) + random.sample(negative_samples, num_negative)
    else:
        return random.sample(pawn_rows, min(sample_size, len(pawn_rows)))

def load_benachmark(paws_path: str = "datasets/final",splits: list = ["train", "dev", "test"]) -> list[PawnRow]:
    # See if the PAWS dataset is available at the specified path, If not, run the download script
    if not os.path.exists(paws_path):
        import subprocess
        import sys

        script_path = os.path.join("datasets", "download_paws.sh")
        if not os.path.exists(script_path):
            print("Download script not found. Please ensure 'download_paws.sh' is in the correct directory.")
            sys.exit(1)

        print("Downloading PAWS dataset...")
        subprocess.run(["bash", script_path], check=True)

    paws_data = []
    for split in splits:
        split_path = os.path.join(paws_path, f"{split}.tsv")
        if not os.path.exists(split_path):
            print(f"Split file {split_path} does not exist. Please check the dataset path.")
            continue

        df = pd.read_csv(split_path, sep="\t", header=None, names=["id", "sentence1", "sentence2", "label"])
        for _, row in df.iterrows():
            paws_data.append(PawnRow(
                id=row["id"],
                sentence1=row["sentence1"],
                sentence2=row["sentence2"],
                label=row["label"]
            ))   
    return paws_data

def add_to_cache(sentence: str, server_base_url: str = "http://localhost:8000"):
    """
    Add a sentence to the cache.
    """
    mock_answer = f"{MOCK_ANSWER_PREFIX}{sentence}"
    put(sentence, mock_answer)
    # response = requests.post(f"{server_base_url}/put", json={"prompt": sentence, "answer": mock_answer})
    # if response.status_code == 200:
    #     print(f"Added to cache: {sentence}")
    # else:
    #     print(f"Failed to add to cache: {response.status_code} - {response.text}")

def get_from_cache(sentence: str, server_base_url: str = "http://localhost:8000") -> str | None:
    """
    Get a sentence from the cache.
    Return string if cache hit, None if not found.
    """
    result = get(sentence)
    if result:
        return result
    else:
        return None
    # response = requests.post(f"{server_base_url}/get", json={"prompt": sentence})
    # if response.status_code == 200:
    #     data = response.json()
    #     cached_answer = data["answer"]
    #     return cached_answer
    # else:
    #     print(f"Failed to retrieve from cache: {response.status_code} - {response.text}")
    #     return None

def query_gptcache(list_of_items: list[PawnRow],gptcache_config_path: str = "cache_config_template.yml",run_server: bool = True, garbage_rate: float = 0.95):
    # Spin up the gptcache with the provided configuration. Run it as a subprocess
    import sys
    if not os.path.exists(gptcache_config_path):
        print(f"GPTCache configuration file {gptcache_config_path} does not exist.")
        sys.exit(1)
    init_similar_cache_from_config(gptcache_config_path)

    eval_items = []
    items_added = []
    # Added garbage to cache
    num_to_add = len(list_of_items)
    for i, item in enumerate(list_of_items):

        if random.random() > garbage_rate:
            add_to_cache(item.sentence1)
            items_added.append(item)
        else:
            add_to_cache(f"garbage_{i}")

    print("Cache populated with sentence 1.")
    # Suffle the items to simulate random access
    # random.shuffle(list_of_items)
    for item in items_added:


        cached_answer = get_from_cache(item.sentence2)
        eval_items.append(PawnRowEval(id=item.id,cached=cached_answer))

    return eval_items


def evaluate_results(eval_items: list[PawnRowEval]):
    """Evaluate the results of the  paws cache evaluation."""
    total_items = len(eval_items)
    cache_hits = sum((1 for eval_item in eval_items if eval_item.cached))
    cache_hit_ratio = cache_hits / total_items

    print(f"Hit rate {cache_hit_ratio:.2f}")

    print(f"Total items: {total_items}")
    


def main():
    args = argparse.ArgumentParser(description="Run PAWS benchmark with GPTCache")
    args.add_argument("--cache-config-path", type=str,  help="Path to the GPTCache configuration file", default="cache_config_template.yml") # baseline_config.yaml # cache_config_template.yml
    args.add_argument("--sample-size", type=int, default=100, help="Number of samples to evaluate from the PAWS dataset")
    args = args.parse_args()
    pawns_items = load_benachmark()
    print(f"Loaded {len(pawns_items)} items from the PAWS benchmark.")
    sample_size = args.sample_size
    pawns_items = choose_samples(pawns_items, sample_size=sample_size, balanced=True)
    print(f"Chosen {len(pawns_items)} samples for evaluation.")
    eval_results = query_gptcache(pawns_items, gptcache_config_path=args.cache_config_path,run_server=False)
    evaluate_results(eval_results)





if __name__ == "__main__":
    main()


