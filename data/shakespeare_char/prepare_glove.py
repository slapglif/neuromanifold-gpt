import numpy as np
import torch
import os
import pickle
from scipy import sparse
from scipy.sparse import linalg


def compute_cooccurrence_embeddings(
    data_path="data/shakespeare_char/train.bin",
    meta_path="data/shakespeare_char/meta.pkl",
    vocab_size=65,
    embedding_dim=128,
    window_size=10,
    save_path="data/shakespeare_char/glove_init.pt",
):
    print(f"Loading data from {data_path}...")
    # Load raw tokens
    data = np.fromfile(data_path, dtype=np.uint16)
    n_tokens = len(data)
    print(f"Dataset size: {n_tokens} tokens")

    # Build co-occurrence matrix
    print("Building co-occurrence matrix...")
    # Use sparse matrix for efficiency
    cooc = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float32)

    # Simple window scan
    # This is O(N * window_size) but N~1M so it's fast enough for Python loop if careful
    # For speed, we can vectorize by shifting

    # We'll use a weighting function 1/d
    for d in range(1, window_size + 1):
        # Shifted arrays
        # Left context: token[i], context[i-d]
        # Right context: token[i], context[i+d]

        # Current tokens
        tokens = data[:-d]
        # Context tokens (future)
        context = data[d:]

        # Update matrix: cooc[tokens, context] += 1/d
        # Doing this via unbuffered add at indices
        # Since we can't easily vectorize sparse updates in numpy without repetition issues,
        # we'll accumulate in a dense 2D histogram if vocab is small (65x65 is tiny!)
        pass

    # Since 65x65 is trivial, use dense matrix
    dense_cooc = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for d in range(1, window_size + 1):
        weight = 1.0 / d
        # Right context
        # data[i] and data[i+d]
        pairs = np.stack([data[:-d], data[d:]], axis=1)
        # Fast 2D histogram
        # Flatten index: row * vocab + col
        flat_indices = pairs[:, 0] * vocab_size + pairs[:, 1]
        counts = np.bincount(flat_indices, minlength=vocab_size * vocab_size)
        dense_cooc += counts.reshape(vocab_size, vocab_size) * weight

        # Symmetric (Left context is same pairs but swapped)
        dense_cooc += counts.reshape(vocab_size, vocab_size).T * weight

    print("Co-occurrence matrix built.")

    # GloVe-style factorization: log(X) = W @ W.T + b
    # Or simpler: PPMI + SVD

    # Add smoothing
    dense_cooc += 1.0

    # Log counts
    log_cooc = np.log(dense_cooc)

    # k must be < min(shape)
    k = min(embedding_dim, vocab_size - 1)
    print(f"Computing SVD with k={k}...")

    # SVD
    U, S, Vh = linalg.svds(log_cooc, k=k)

    # Embeddings = U * sqrt(S)
    embeddings = U @ np.diag(np.sqrt(S))

    # If k < embedding_dim, pad with noise
    if k < embedding_dim:
        pad_dim = embedding_dim - k
        padding = np.random.normal(0, 0.02, (vocab_size, pad_dim))
        embeddings = np.concatenate([embeddings, padding], axis=1)

    # Convert to torch
    emb_tensor = torch.from_numpy(embeddings).float()

    # Normalize
    emb_tensor = torch.nn.functional.normalize(emb_tensor, dim=1)

    print(f"Embeddings shape: {emb_tensor.shape}")
    print(f"Saving to {save_path}")
    torch.save(emb_tensor, save_path)
    return emb_tensor


if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists("data/shakespeare_char/train.bin"):
        print("Data not found. Run prepare.py first.")
    else:
        # Load meta to get true vocab size
        with open("data/shakespeare_char/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        v_size = meta["vocab_size"]

        compute_cooccurrence_embeddings(
            vocab_size=v_size,
            embedding_dim=128,  # Match model config
            window_size=10,
        )
