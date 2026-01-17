import torch
import math

def check_continuity(vocab_size, path_length):
    base = int(math.ceil(vocab_size ** (1 / path_length)))
    indices = torch.arange(vocab_size)
    gray = indices ^ (indices >> 1)
    
    token_map = torch.zeros(vocab_size, path_length, dtype=torch.long)
    temp = gray.clone()
    for i in range(path_length):
        token_map[:, path_length - 1 - i] = temp % base
        temp //= base
    
    # Calculate differences between adjacent token words
    diffs = (token_map[1:] != token_map[:-1]).sum(dim=1)
    
    max_diff = diffs.max().item()
    mean_diff = diffs.float().mean().item()
    jumps = (diffs > 1).sum().item()
    
    print(f"Base: {base}")
    print(f"Max generator changes between adjacent tokens: {max_diff}")
    print(f"Mean generator changes: {mean_diff:.4f}")
    print(f"Number of tokens where >1 generator changes: {jumps} / {vocab_size-1}")

if __name__ == "__main__":
    check_continuity(150000, 3)
