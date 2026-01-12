# neuromanifold_gpt/model/attention/knot.py
"""
Knot-Theoretic Attention (Topological Gating).

Uses braid group / knot invariants to identify structural patterns.
Treats token sequences as strands in a 3D manifold.
Computes "linking numbers" to gate attention: only "entangled" tokens attend.

Reference: MANIFOLD_ATTENTION_ARCHITECTURE.md Section 4
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnotAttention(nn.Module):
    """
    Attention based on topological invariants of token "braids".

    Concept: Treat sequence as strands that can cross over/under.
    Crossings encode relationships; invariants identify patterns.
    """

    def __init__(self, embed_dim: int, manifold_dim: int = 64, n_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.manifold_dim = manifold_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Learn crossing detector: which tokens "cross" (strongly interact)
        # Input: features_i, features_j, coords_i, coords_j
        input_dim = (embed_dim // n_heads) * 2 + manifold_dim * 2
        
        self.crossing_net = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 2)  # [over, under] scores
        )

        # Alexander polynomial coefficients (knot invariant features)
        # Maps attended features to topological features
        self.invariant_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Projections
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def compute_writhe(self, crossings: torch.Tensor) -> torch.Tensor:
        """
        Writhe = sum of signed crossings.
        Topological invariant indicating "twist" of the braid.
        crossings: (B, H, T, T, 2)
        """
        # score[0] = over, score[1] = under
        # signed crossing = over - under
        signed = crossings[..., 0] - crossings[..., 1]
        return signed.sum(dim=[-1, -2])  # (B, H)

    def compute_linking_matrix(self, crossings: torch.Tensor) -> torch.Tensor:
        """
        Linking number matrix between strands i and j.
        How many times they wind around each other.
        """
        # Simplified: net crossing score
        return crossings[..., 0] - crossings[..., 1]

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, T, D) token features
            coords: (B, T, M) manifold coordinates

        Returns:
            out: (B, T, D)
            info: dict
        """
        B, T, D = x.shape
        H = self.n_heads
        HD = self.head_dim

        # Project features
        q = self.to_q(x).view(B, T, H, HD).transpose(1, 2) # (B, H, T, HD)
        k = self.to_k(x).view(B, T, H, HD).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, HD).transpose(1, 2)

        # Prepare pairwise inputs for crossing net
        # This is O(T^2), but T is usually block_size (256-1024)
        # Optimization: Could use block-sparse or local window + random long-range
        
        # Expand for pairwise: (B, H, T, T, HD*2 + M*2)
        # We process heads in parallel but share crossing net weights (or could group)
        # To save memory, we might need to chunk or loop if T is large. 
        # For T=256, T^2=65k, acceptable.
        
        # Let's use the Q and K as the feature proxies for "crossing prediction"
        # q_i: (B, H, T, 1, HD)
        # k_j: (B, H, 1, T, HD)
        q_exp = q.unsqueeze(3).expand(-1, -1, -1, T, -1)
        k_exp = k.unsqueeze(2).expand(-1, -1, T, -1, -1)
        
        # Coords: (B, T, M) -> (B, 1, T, M) -> expand
        c_i = coords.unsqueeze(1).unsqueeze(3).expand(-1, H, -1, T, -1)
        c_j = coords.unsqueeze(1).unsqueeze(2).expand(-1, H, T, -1, -1)
        
        # Concatenate: (B, H, T, T, InputDim)
        # Warning: This tensor can be large. 
        # For B=4, H=4, T=256, D=128 (HD=32), M=32
        # InputDim = 64 + 64 = 128
        # Size = 4 * 4 * 256 * 256 * 128 * 4 bytes ~= 536 MB. OK for GPU.
        pair_input = torch.cat([q_exp, k_exp, c_i, c_j], dim=-1)
        
        # Predict crossings
        # (B, H, T, T, 2)
        crossings = self.crossing_net(pair_input)
        crossings = F.softmax(crossings, dim=-1)

        # Topological Linking Matrix
        # (B, H, T, T)
        linking = self.compute_linking_matrix(crossings)
        
        # Topological Gating (Masking)
        # If linking number is close to 0, they are unlinked -> low attention
        # We use the absolute linking magnitude as the attention score
        # Note: Standard attention uses dot product similarity.
        # Here we substitute (or augment) dot product with "entanglement amount"
        
        # Scale for softmax stability
        attn_scores = linking / math.sqrt(self.head_dim)
        
        # Causal mask (if autoregressive)
        # Ideally, knots are global, but GPT generation is causal.
        # We mask future tokens to prevent cheating.
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply to values
        # (B, H, T, T) @ (B, H, T, HD) -> (B, H, T, HD)
        out = torch.matmul(attn_weights, v)
        
        # Apply Invariant Transformation (Alexander Polynomial features)
        # Reshape to (B, T, D) first
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.invariant_net(out)
        
        # Final projection
        out = self.to_out(out)
        
        # Compute auxiliaries
        writhe = self.compute_writhe(crossings)

        info = {
            'knot_writhe': writhe,
            'knot_linking_mean': linking.abs().mean()
        }

        return out, info
