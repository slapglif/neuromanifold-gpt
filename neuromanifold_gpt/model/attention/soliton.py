# neuromanifold_gpt/model/attention/soliton.py
"""
Soliton Attention based on Kaufmann-Heimburg Model.

Action potentials as acoustic solitons in lipid membranes (NOT electrical).
Key properties:
- Threshold behavior (all-or-none)
- Stable wave propagation
- Collision without annihilation

Reference: https://www.nbi.ku.dk/membranes/Kaufmann/publications.html
"""
import torch
import torch.nn as nn
from einops import rearrange, einsum


class SolitonDynamics(nn.Module):
    """
    Simplified Fitzhugh-Nagumo + Heimburg-Jackson soliton dynamics.

    dv/dt = v - v^3/3 - w + I
    dw/dt = (v + a - bw) / tau

    Where v is membrane potential and w is recovery variable.
    """

    def __init__(self, dim: int, tau: float = 0.1, threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.threshold = threshold

        # Learnable parameters
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.8))


    def forward(self, stimulus: torch.Tensor, n_steps: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve soliton dynamics.

        Args:
            stimulus: (B, T, D) input current
            n_steps: integration steps

        Returns:
            response: (B, T, D) soliton response
            state: (B, T, D) final membrane state
        """
        dt = 1.0 / n_steps

        # Initialize state
        v = torch.zeros_like(stimulus)
        w = torch.zeros_like(stimulus)

        # Normalize stimulus magnitude to prevent numerical explosion
        # The FitzHugh-Nagumo model is stable for |I| < ~1.5
        stim_scale = stimulus.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        stimulus_normed = stimulus / stim_scale

        # Apply threshold behavior (all-or-none response)
        # Above threshold: full normalized input
        # Below threshold: attenuated (10x weaker)
        I = torch.where(
            stimulus.abs() > self.threshold,
            stimulus_normed,
            stimulus_normed * 0.1  # Sub-threshold attenuation
        )

        # Integrate dynamics with stability clamps
        for _ in range(n_steps):
            dv = v - v.pow(3) / 3 - w + I
            dw = (v + self.a - self.b * w) / self.tau

            v = (v + dt * dv).clamp(-3.0, 3.0)
            w = (w + dt * dw).clamp(-3.0, 3.0)

        # Scale response by original stimulus magnitude
        # This preserves the threshold effect while being numerically stable
        response = v * stim_scale

        return response, v


class SolitonAttention(nn.Module):
    """
    Attention via soliton wave propagation.

    Instead of softmax attention, attention "propagates" as stable
    soliton waves across the manifold. Information travels in
    wave packets with characteristic width and velocity.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        threshold: float = 0.5,
        tau: float = 0.1,
        pulse_width_base: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.threshold = threshold
        self.pulse_width_base = pulse_width_base

        assert embed_dim % n_heads == 0

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Soliton dynamics per head
        self.soliton = SolitonDynamics(self.head_dim, tau, threshold)

        # Pulse width modulation (content-dependent)
        self.pulse_width_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.SiLU(),
            nn.Linear(self.head_dim // 2, 1),
            nn.Softplus()
        )

        # Spectral filtering
        self.spectral_filter = nn.Parameter(torch.ones(n_heads, 32) * 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        spectral_basis: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Soliton attention forward pass.

        Args:
            x: (B, T, D) input
            spectral_basis: (B, T, k) eigenvectors

        Returns:
            out: (B, T, D) attention output
            info: dict with soliton diagnostics
        """
        B, T, D = x.shape
        k = spectral_basis.shape[-1]

        # QKV projection
        qkv = self.qkv(x)
        q, key, v = rearrange(qkv, 'b t (three h d) -> three b h t d', three=3, h=self.n_heads)

        # Compute pulse widths (content-dependent)
        pulse_widths = self.pulse_width_net(q.mean(dim=2))  # (B, H, 1)
        pulse_widths = self.pulse_width_base + pulse_widths.squeeze(-1)  # (B, H)

        # Project to spectral domain
        basis = spectral_basis.unsqueeze(1)  # (B, 1, T, k)
        q_spec = einsum(q, basis, 'b h t d, b one t k -> b h k d')
        k_spec = einsum(key, basis, 'b h t d, b one t k -> b h k d')

        # Spectral attention weights (pre-soliton)
        # Dot product over head dimension: sum_d(q_d * k_d)
        attn_spec = einsum(q_spec, k_spec, 'b h k d, b h k d -> b h k')
        attn_spec = attn_spec / (self.head_dim ** 0.5)

        # Apply spectral filter (frequency-dependent)
        filter_weights = torch.sigmoid(self.spectral_filter[:, :k])  # (H, k)
        attn_spec = attn_spec * filter_weights.unsqueeze(0)

        # Pass through soliton dynamics (threshold + wave propagation)
        soliton_response, soliton_state = self.soliton(
            attn_spec.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )
        soliton_response = soliton_response.mean(dim=-1)  # (B, H, k)

        # Apply to values in spectral domain
        v_spec = einsum(v, basis, 'b h t d, b one t k -> b h k d')
        out_spec = einsum(soliton_response.unsqueeze(-1), v_spec, 'b h k one, b h k d -> b h k d')

        # Project back to spatial domain
        out = einsum(out_spec, basis, 'b h k d, b one t k -> b h t d')
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.dropout(self.out_proj(out))

        info = {
            'pulse_widths': pulse_widths,
            'soliton_state': soliton_state.mean(),
            'spectral_response': soliton_response
        }

        return out, info
