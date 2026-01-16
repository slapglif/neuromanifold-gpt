
import torch
import torch.nn as nn
from neuromanifold_gpt.model.attention.fhn import FHNAttention
from neuromanifold_gpt.model.attention.knot import KnotAttention
from neuromanifold_gpt.model.embeddings.ramanujan import RamanujanPositionalEmbedding


def _kaufmann_reaction_diffusion_step(
    u: torch.Tensor,
    w: torch.Tensor,
    I_base: torch.Tensor,
    diffused_signal: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float,
    dt: torch.Tensor,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kaufmann reaction-diffusion update loop.
    Combines FHN dynamics with topological diffusion.

    This function is compiled with torch.compile for kernel fusion.
    """
    # Precompute diffusion input (constant across iterations)
    I_diff = torch.tanh(diffused_signal)

    for _ in range(n_steps):
        # A. Reaction (FHN Local Dynamics)
        # Safe clamping to prevent numerical explosion
        u_safe = u.clamp(-2.0, 2.0)

        # FHN cubic nonlinearity
        du = u_safe - (u_safe**3) / 3.0 - w

        # Add External Input (Stimulus + Diffusion)
        drive = I_base + I_diff
        du = du + drive

        # Update u
        u_new = u + dt * du

        # Update w (Recovery variable)
        dw = (u_safe + a - b * w) / tau
        w_new = w + dt * dw

        # Final safety clamp
        u = u_new.clamp(-3.0, 3.0)
        w = w_new.clamp(-3.0, 3.0)

    return u, w


# Compile with reduce-overhead mode for minimal Python overhead
# Gracefully fall back to uncompiled version on Python 3.12+ (where Dynamo is not supported)
try:
    kaufmann_reaction_diffusion_step = torch.compile(
        _kaufmann_reaction_diffusion_step,
        mode="reduce-overhead"
    )
except RuntimeError as e:
    if "Dynamo is not supported" in str(e):
        # Fall back to uncompiled version on Python 3.12+
        kaufmann_reaction_diffusion_step = _kaufmann_reaction_diffusion_step
    else:
        raise


class KaufmannAttention(nn.Module):
    """
    The Kaufmann Trifecta Attention Model (V2: Reaction-Diffusion).
    
    Synthesizes:
    1. Konrad Kaufmann: Soliton propagation (FHN Reaction)
    2. Stuart Kauffman: Fitness landscape (WaveKAN modulation)
    3. Louis Kauffman: Topological Linking (Knot Diffusion)
    
    Mechanism:
    Reaction-Diffusion on the Token Graph.
    - Nodes = Tokens.
    - Edges = Knot Topological Links (Attention Matrix).
    - Dynamics = FHN (Excitable Medium).
    
    Equation:
    du/dt = FHN(u) + Diffusion(u, KnotLinks)
    """
    def __init__(self, embed_dim, n_heads, config=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        # Extract config
        manifold_dim = getattr(config, 'manifold_dim', kwargs.get('manifold_dim', 64))
        fhn_threshold = getattr(config, 'fhn_threshold', kwargs.get('fhn_threshold', 0.5))
        fhn_tau = getattr(config, 'fhn_tau', kwargs.get('fhn_tau', 12.5))
        self.n_fhn_steps = getattr(config, 'n_fhn_steps', kwargs.get('n_fhn_steps', 2))
        pos_emb_type = getattr(config, 'pos_emb_type', kwargs.get('pos_emb_type', 'learned'))
        max_seq_len = getattr(config, 'max_seq_len', kwargs.get('max_seq_len', 1024))

        # 1. Topology (Diffusion Matrix)
        # Computes 'Linking Number' matrix -> Attention Weights
        self.knot_gate = KnotAttention(embed_dim, manifold_dim=manifold_dim, n_heads=n_heads, pos_emb_type=pos_emb_type, max_seq_len=max_seq_len)
        
        # 2. Dynamics (Reaction)
        # FHN parameters
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.8))
        self.dt = nn.Parameter(torch.tensor(0.1))
        self.threshold = fhn_threshold
        self.tau = fhn_tau
        
        # Diffusion rate (coupling strength)
        self.diffusion_rate = nn.Parameter(torch.tensor(0.1))
        
        # Mixing
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, spectral_basis=None):
        """
        Args:
            x: (B, T, D) Input features
            spectral_basis: Optional (can be coords if provided, otherwise create from x)
        """
        B, T, D = x.shape
        H = self.n_heads

        # Use spectral_basis as coords if provided, otherwise create proxy
        coords = spectral_basis if spectral_basis is not None else None

        # 1. Compute Topology (Knot Attention)
        diffused_signal, knot_info = self.knot_gate(x, coords)
        
        # Normalize input to prevent shock
        # FHN likes inputs in [-1, 1] approx
        I_base = torch.tanh(x) 
        
        # 2. Initialize State (u, w)
        u = I_base
        w = torch.zeros_like(u)

        # 3. Reaction-Diffusion Loop
        # Clamp dt to safe range [0.01, 0.2]
        dt = torch.sigmoid(self.dt) * 0.2

        # Execute dynamics using torch.compile-optimized kernel
        u, w = kaufmann_reaction_diffusion_step(
            u, w, I_base, diffused_signal, self.a, self.b, self.tau, dt, self.n_fhn_steps
        )
            
        # 4. Output
        out = self.out_proj(u)

        # Update info dict with correct attention type
        knot_info['attention_type'] = 'kaufmann'

        return out, knot_info
