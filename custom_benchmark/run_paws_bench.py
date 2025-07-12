import os
from dataclasses import dataclass
from typing import Optional, Union
import pandas as pd
import time
@dataclass
class PawnRow:
    id: Optional[str] = None
    sentence1: Optional[str] = None
    sentence2: Optional[str] = None
    label: Optional[Union[int, str]] = None

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

def query_gptcache(list_of_items: list[PawnRow],gptcache_config_path: str = "cache_config_template.yml"):
    # Spin up the gptcache with the provided configuration. Run it as a subprocess
    import subprocess
    import sys
    if not os.path.exists(gptcache_config_path):
        print(f"GPTCache configuration file {gptcache_config_path} does not exist.")
        sys.exit(1)

    print("Starting GPTCache with the provided configuration...")
    try:
        server_path = os.path.join("gptcache_server", "server.py")
        proc = subprocess.Popen(["python", server_path, "--cache-config-file", gptcache_config_path])
    except subprocess.CalledProcessError as e:
        print(f"Failed to start GPTCache server: {e}")
        sys.exit(1)
    try:
        time.sleep(20)  # Server startup time, adjust as necessary

        # ---- YOUR CODE TO QUERY THE SERVER HERE ----
        # e.g. for item in list_of_items: ... 
        # Do your requests using the running server

        # Example placeholder
        print("Server running, perform your queries here...")
        time.sleep(1)

    finally:
        print("Shutting down GPTCache server...")
        # Graceful shutdown: send SIGTERM (or use your server's shutdown endpoint if available)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("Force killing the GPTCache server...")
            proc.kill()
        print("Server stopped.")    

def evaluate_results():
    pass


def main():
    pawns_items = load_benachmark()
    print(f"Loaded {len(pawns_items)} items from the PAWS benchmark.")
    query_gptcache(pawns_items)






if __name__ == "__main__":
    main()