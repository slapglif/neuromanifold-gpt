"""
Prepare the OpenWebText dataset for training.

Uses the Hugging Face datasets library to download and tokenize.
Creates train.bin and val.bin files with GPT-2 BPE tokens.

Usage:
    python neuromanifold_gpt/data/prepare_openwebtext.py

Requirements:
    pip install datasets tiktoken tqdm
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data/openwebtext"
NUM_PROC = 8  # Number of workers for tokenization


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load dataset
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", num_proc=NUM_PROC)

    # Get train/val split
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005,
        seed=2357,
        shuffle=True,
    )
    split_dataset["val"] = split_dataset.pop("test")

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    # Tokenize
    print("Tokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )

    # Write to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(OUT_DIR, f"{split}.bin")
        print(f"Writing {filename} ({arr_len:,} tokens)...")

        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"Writing {split}"):
            # Shard the dataset into batches
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Save metadata (vocab size is GPT-2's 50257 but we round to 50304)
    import pickle

    meta = {
        "vocab_size": 50304,  # GPT-2 vocab rounded up
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Data saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
