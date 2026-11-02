import torch
import pickle
import os
import sys
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    pass


from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.model.wave_manifold_gpt import WaveManifoldGPT


def generate_text():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "data/shakespeare_char"
    meta_path = os.path.join(data_dir, "meta.pkl")

    if not os.path.exists(meta_path):
        print("Meta file not found")
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    decode = lambda l: "".join([itos[i] for i in l])
    encode = lambda s: [stoi[c] for c in s]
    vocab_size = meta["vocab_size"]  # Should be 65

    # Reconstruct Config - Matching training
    config = WaveManifoldConfig(
        vocab_size=256,
        block_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
        use_fno_encoder=True,
        fno_modes=16,
        backbone_type="hyena",
        use_mamba_backbone=False,
        use_soliton_mixing=True,
        soliton_type="sine_gordon",
        use_topological_loss=False,
        use_continuous_head=False,
    )

    model = WaveManifoldGPT(config).to(device)

    # Load Checkpoint
    ckpt_path = "out-wave-shakespeare/last-v2.ckpt"
    if not os.path.exists(ckpt_path):
        # Fallback if V2 doesn't exist
        ckpt_path = "out-wave-shakespeare/last.ckpt"

    print(f"Loading from {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        # Clean keys
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[6:] if k.startswith("model.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("Loaded weights.")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    model.eval()

    # Generate
    start_str = "\n"
    start_ids = encode(start_str)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    print("\n--- GENERATING SHAKESPEARE (Semantic Wave) ---\n")

    max_new_tokens = 500
    temperature = 0.8
    top_k = 65

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = (
                x if x.size(1) <= config.block_size else x[:, -config.block_size :]
            )

            logits, _, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Filter invalid tokens
            if logits.size(-1) > vocab_size:
                logits[:, vocab_size:] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, idx_next], dim=1)

            try:
                char = decode([idx_next.item()])
                print(char, end="", flush=True)
            except:
                pass

    print("\n\n--- END ---")


if __name__ == "__main__":
    generate_text()
