"""
Prepare the Shakespeare dataset for character-level training.

Creates train.bin and val.bin files with tokenized data.

Usage:
    python neuromanifold_gpt/data/prepare_shakespeare.py
"""
import os
import pickle
import urllib.request

import numpy as np

# Download Shakespeare text
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
INPUT_FILE = "input.txt"
OUT_DIR = "data/shakespeare_char"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Download if needed
    input_path = os.path.join(OUT_DIR, INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, input_path)

    # Load text
    with open(input_path, encoding="utf-8") as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")

    # Build character vocabulary
    chars = sorted(set(data))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode entire dataset
    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    # Train/val split (90/10)
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Save to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Data saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
