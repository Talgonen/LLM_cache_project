import os
from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
from custom_benchmark.common import MOCK_ANSWER_PREFIX
import random
import argparse
from pathlib import Path
from gptcache.adapter.api import (
    get,
    put,
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


def query_gptcache(list_of_items: list[PawnRow],gptcache_config_path: str = "cache_config_template.yml", garbage_rate: float = 0.95):
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
    
    return {
        "cache_hit_ratio": cache_hit_ratio,
        "total_items": total_items,
        "cache_hits": cache_hits
    }



def main():
    args = argparse.ArgumentParser(description="Run PAWS benchmark with GPTCache")
    args.add_argument("--cache-config-path", type=str,  help="Path to the GPTCache configuration file", default="cache_config_template.yml") # baseline_config.yaml # cache_config_template.yml
    args.add_argument("--sample-size", type=int, default=100, help="Number of samples to evaluate from the PAWS dataset")
    args.add_argument("--garbage_rates", type=list[float], nargs="+", default=[0.95], help="Garbage rate to use for the evaluation (default: [0.95])")
    args = args.parse_args()
    pawns_items = load_benachmark()

    all_results = []
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)
    sample_size = args.sample_size
    garbage_rates = args.garbage_rates
    save_path = f"results/{args.cache_config_path}_garbage.csv"
    print(f"Loaded {len(pawns_items)} items from the PAWS benchmark.")
    print(f"Sample size: {sample_size}, Garbage rates: {garbage_rates}")

    for garbage_rate in garbage_rates:
        print(f"Running evaluation with garbage rate: {garbage_rate}")
        pawns_items = choose_samples(pawns_items, sample_size=sample_size, balanced=True)
        eval_results = query_gptcache(pawns_items, gptcache_config_path=args.cache_config_path, garbage_rate=garbage_rate)
        res = evaluate_results(eval_results)
        all_results.append(res)

    print("All evaluations completed.")
    print(all_results)

    # Save results to a file or process further as needed
    # For example, you can save to a CSV file:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(save_path, index=False)
    print(save_path)





if __name__ == "__main__":
    main()


