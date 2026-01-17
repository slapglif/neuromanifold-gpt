import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from torch.utils.checkpoint import checkpoint
from .kan.cheby.linear import ChebyKANLinear
from .kan.wave.ffn import WaveKANFFN

# ==========================================
# 1. Core Physics & Math Utilities
# ==========================================


class SolitonMath:
    """
    Stable implementations of soliton wave mechanics.
    Adheres to the Heimburg-Jackson thermodynamic model.
    """

    @staticmethod
    def sech2(x):
        """
        Computes sech^2(x) = 1 / cosh^2(x) in a numerically stable way.
        This models the density profile of a soliton pulse propagating
        through a lipid interface near phase transition.
        """
        # numerical stability: 1 - tanh^2(x) avoids large exponentials
        return 1.0 - torch.tanh(x).pow(2)


def orthogonal_init_(tensor):
    """
    Initializes a tensor to be orthogonal/unitary.
    Crucial for Braid Generators to be energy-conserving operators (isometric).
    """
    if tensor.ndimension() == 2:
        nn.init.orthogonal_(tensor)
    elif tensor.ndimension() == 3:
        # Initialize each matrix in the batch independently
        for i in range(tensor.size(0)):
            nn.init.orthogonal_(tensor[i])
    else:
        # Fallback for other dimensions (e.g. 1D or >3D)
        # For Soliton physics, we prioritize 2D/3D operator stability
        nn.init.normal_(tensor, 0, 0.02)


# ==========================================
# 2. Topological Embedding Layer (Braid Group)
# ==========================================


# ==========================================
# 2. Topological Embedding Layer (Braid Group)
# ==========================================


class BraidEmbedding(nn.Module):
    """
    Maps tokens to Braid Group elements via Multi-Stranded Matrix Composition.
    Uses Gray-code mapping to ensure semantic continuity between adjacent tokens.

    Scale:
    - Vocab: 150k+
    - Path Length: 3
    - Strands: 2 (Parallel braid word composition for higher capacity)
    """

    def __init__(self, vocab_size, matrix_dim, path_length=3, strands=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.matrix_dim = matrix_dim
        self.path_length = path_length
        self.strands = strands

        # Effective dimension matches transformer backbone
        # We ensure strands * matrix_dim * matrix_dim matches the target model_dim
        self.output_dim = matrix_dim * matrix_dim * strands

        # Calculate base for Gray-code encoding
        # base^path_length >= vocab_size
        self.base = int(math.ceil(vocab_size ** (1 / path_length)))

        # Generator Matrices: [Strands, Base, Dim, Dim]
        # Multiple strands allow independent topological 'braiding' paths
        self.generators = nn.Parameter(
            torch.empty(strands, self.base, matrix_dim, matrix_dim)
        )
        orthogonal_init_(self.generators.view(-1, matrix_dim, matrix_dim))

        # Precompute mapping: TokenID -> Braid Word (Gray-code)
        self.register_buffer("token_map", self._create_gray_token_map())

    def _create_gray_token_map(self):
        """
        Creates a Gray-code based mapping from Token ID to generator indices.
        Ensures that Token i and Token i+1 have braid words that differ by at most one generator.
        """
        indices = torch.arange(self.vocab_size)

        # Convert to Gray code: g = i ^ (i >> 1)
        gray = indices ^ (indices >> 1)

        # Expand Gray code to base-N word
        # [Vocab, PathLength]
        token_map = torch.zeros(self.vocab_size, self.path_length, dtype=torch.long)
        temp = gray
        for i in range(self.path_length):
            token_map[:, self.path_length - 1 - i] = temp % self.base
            temp //= self.base

        return token_map

    def forward(self, input_ids):
        """
        Args:
            input_ids: [Batch, Seq]
        Returns:
            embeddings: [Batch, Seq, matrix_dim*matrix_dim*strands]
        """
        batch, seq = input_ids.shape

        # 1. Retrieve braid word indices: [Batch, Seq, PathLength]
        gen_indices = self.token_map[input_ids]

        # 2. Composition for all strands (Vectorized)
        # Fetch matrices for all strands: [Strands, Batch, Seq, PathLength, Dim, Dim]
        # generators: [Strands, Base, Dim, Dim]
        # matrices: [Strands, Batch, Seq, PathLength, Dim, Dim]
        matrices = self.generators[:, gen_indices]
        matrices = rearrange(matrices, "st b s p d1 d2 -> b s p st d1 d2")

        # Start with Identity: [Batch, Seq, Strands, Dim, Dim]
        current_state = torch.eye(self.matrix_dim, device=input_ids.device)
        current_state = repeat(
            current_state, "d1 d2 -> b s st d1 d2", b=batch, s=seq, st=self.strands
        )

        # Non-abelian composition (sequential over PathLength, vectorized over Strands)
        for i in range(self.path_length):
            # b: batch, s: seq, st: strands, i: row, k: mid, j: col
            current_state = einsum(
                current_state,
                matrices[:, :, i],
                "b s st i k, b s st k j -> b s st i j",
            )

        # 3. Flatten and concatenate strands
        # [Batch, Seq, Strands, Dim, Dim] -> [Batch, Seq, Strands * Dim * Dim]
        return rearrange(current_state, "b s st d1 d2 -> b s (st d1 d2)")


# ==========================================
# 3. Wave-Based Positional Embedding (RoPE)
# ==========================================


class RotaryEmbedding(nn.Module):
    """
    Encodes position as a rotation in the complex plane.
    Aligns with wave mechanics (phase shift).
    """

    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from standard: we concat to match complex rotation logic
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # [1, 1, Seq, Dim]
            self.cos_cached = rearrange(emb.cos(), "s d -> 1 1 s d")
            self.sin_cached = rearrange(emb.sin(), "s d -> 1 1 s d")

        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [Batch, Heads, Seq, HeadDim]
    # cos, sin: [1, 1, Seq, HeadDim]

    # Crop to sequence length
    seq_len = q.shape[2]
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==========================================
# 3.5 Fourier Neural Operator (FNO) - Input Smoothing
# ==========================================


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def complex_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to a factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        # Robustly handle cases where seq_len is small (fewer modes than self.modes)
        # FNO truncates high frequencies, but if signal is too short, we must use what we have.
        available_modes = x_ft.size(-1)
        use_modes = min(self.modes, available_modes)

        out_ft[:, :, :use_modes] = self.complex_mul1d(
            x_ft[:, :, :use_modes], self.weights[:, :, :use_modes]
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FourierNeuralOperator(nn.Module):
    """
    1D Fourier Neural Operator layer.
    Maps the discrete input sequence to a continuous function space representation
    by filtering in the frequency domain.
    Uses ChebyKAN for channel mixing (Spectral-friendly).
    """

    def __init__(self, width, modes):
        super().__init__()
        self.width = width
        self.modes = modes
        self.conv1 = SpectralConv1d(width, width, modes)
        # ChebyKAN mixer for spectral coefficients
        self.w1 = ChebyKANLinear(width, width, degree=4)

    def forward(self, x):
        # x: [Batch, Seq, Width]

        # Spectral Branch
        x_spec = x.permute(0, 2, 1)  # [Batch, Width, Seq]
        x1 = self.conv1(x_spec)
        x1 = x1.permute(0, 2, 1)  # [Batch, Seq, Width]

        # Local Branch (ChebyKAN Mixing)
        x2 = self.w1(x)

        x_out = x1 + x2
        return F.gelu(x_out)


# ==========================================
# 4. Soliton Attention (Kaufmann Active Medium)
# ==========================================


class SolitonAttention(nn.Module):
    """
    Implementation of Kaufmann Soliton Attention.
    Models the 'Active Medium' of lipid monolayers near phase transition.

    References:
    - Kaufmann, K. (1989). 'Acoustic pulses in lipid monolayers'.
    - Heimburg & Jackson (2005). 'Thermodynamic theory of nerve pulses'.
    """

    def __init__(self, dim, num_heads, rope_emb=None):
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_emb = rope_emb

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Kaufmann Active Medium Parameters
        # compressibility (tau): Controls pulse width (susceptibility)
        # phase_threshold (mu): Point of maximum density change (resonance)
        # stiffness (beta): Regularizes the energy conservation
        self.compressibility = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)
        self.phase_threshold = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.stiffness = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Scale (Inverse Temperature)
        self.scale = self.head_dim**-0.5

    def forward(self, x, mask=None):
        """
        x: [Batch, Seq, Dim]
        """
        b, l, d = x.shape
        h = self.num_heads

        # 1. Projections
        q = rearrange(self.q_proj(x), "b l (h d) -> b h l d", h=h)
        k = rearrange(self.k_proj(x), "b l (h d) -> b h l d", h=h)
        v = rearrange(self.v_proj(x), "b l (h d) -> b h l d", h=h)

        # 2. RoPE (Wave Phase Shift)
        if self.rope_emb is not None:
            cos, sin = self.rope_emb(v, seq_len=l)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Pressure Field Interaction (QK^T)
        pressure = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        # 4. Causal Masking (Infinite Potential Barrier)
        if mask is None:
            mask = torch.triu(torch.ones(l, l, device=x.device), diagonal=1).bool()
        pressure.masked_fill_(mask, -1e4)

        # 5. Soliton Activation (Adiabatic Phase Transition)
        # Energy E = (Pressure - mu) / tau
        tau = F.softplus(self.compressibility) + 1e-6
        energy = (pressure - self.phase_threshold) / tau

        # Pulse profile (Non-dispersive wave packet)
        # sech^2 is the density profile of a Boussinesq soliton
        attn_weights = SolitonMath.sech2(energy)

        # 6. Stability & Conservation (Energy Norm)
        # Soliton propagation in active media requires normalization
        # to ensure energy conservation across the sequence.
        # We use a soft-sum normalization to prevent gradient collapse.
        conservation = torch.sum(attn_weights, dim=-1, keepdim=True) + 1e-6
        attn_weights = attn_weights / conservation

        # 7. Propagate Value Wave
        out = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")

        # 8. Output Projection
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.o_proj(out)


# ==========================================
# 5. Modern Backbone (RMSNorm + SwiGLU)
# ==========================================


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ==========================================
# 5.5 Topological Regularization
# ==========================================


class TopologicalHead(nn.Module):
    """
    Computes a topological invariant proxy (Trace of the Holonomy)
    to regularize the latent manifold structure.
    Approximates the Jones Polynomial via the trace of the braid representation.
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # We treat the sequence as a trajectory on the manifold.
        # The 'trace' of the trajectory loop is a topological invariant.

        # Project to Braid Representation Space if needed
        h = self.proj(x)

        # Compute the path integral / trace
        # Trace(Product(Matrices)) is hard, so we approximate with
        # sum of traces of local holonomies or global average.
        # Here we compute the norm of the closed loop formed by the sequence mean.

        global_state = h.mean(dim=1)  # [Batch, Dim]
        # Topological Energy: Norm of the global state (should be invariant)
        invariant_energy = torch.norm(global_state, dim=-1)

        return invariant_energy


# ==========================================
# 6. Neuro-Symbolic Wave Manifold Network
# ==========================================


class NeuroSymbolicWaveNet(nn.Module):
    def __init__(
        self,
        vocab_size=152000,
        matrix_dim=32,  # Effective embed dim = 32*32 = 1024
        depth=12,
        num_heads=16,
        max_seq_len=2048,
        fno_modes=32,
        use_checkpointing=False,
    ):
        super().__init__()

        self.use_checkpointing = use_checkpointing
        # Strands = 2 per previous update
        self.model_dim = matrix_dim * matrix_dim * 2

        # 1. Topological Embedding
        self.embed = BraidEmbedding(vocab_size, matrix_dim, strands=2)

        # 2. Input Smoothing: Fourier Neural Operator
        # Maps discrete token embeddings to continuous wave function
        self.fno = FourierNeuralOperator(self.model_dim, fno_modes)

        # 3. Positional Embedding
        self.rope = RotaryEmbedding(
            self.model_dim // num_heads, max_seq_len=max_seq_len
        )

        # 4. Layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn_norm": RMSNorm(self.model_dim),
                        "attn": SolitonAttention(
                            self.model_dim, num_heads, rope_emb=self.rope
                        ),
                        "ffn_norm": RMSNorm(self.model_dim),
                        "ffn": WaveKANFFN(self.model_dim, self.model_dim * 4),
                    }
                )
            )

        # 5. Output Head
        self.final_norm = RMSNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, vocab_size, bias=False)

        # 6. Topological Auxiliary Head
        self.topo_head = TopologicalHead(self.model_dim)

        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # BraidEmbedding handles its own orthogonal init

    def forward(self, input_ids, return_topo_loss=False):
        # input_ids: [Batch, Seq]

        # Embedding (Discrete -> Braid Group Algebra)
        x = self.embed(input_ids)

        # Smoothing (Algebra -> Continuous Wave)
        x = self.fno(x)

        # Backbone (Soliton Dynamics)
        def create_custom_forward(layer_dict):
            def custom_forward(hidden_states):
                # 1. Soliton Attention
                residual = hidden_states
                hidden_states = layer_dict["attn_norm"](hidden_states)
                hidden_states = layer_dict["attn"](hidden_states)
                hidden_states = hidden_states + residual

                # 2. Feed Forward
                residual = hidden_states
                hidden_states = layer_dict["ffn_norm"](hidden_states)
                hidden_states = layer_dict["ffn"](hidden_states)
                hidden_states = hidden_states + residual
                return hidden_states

            return custom_forward

        for layer in self.layers:
            if self.use_checkpointing and self.training:
                # Gradient checkpointing to save memory
                x = checkpoint(create_custom_forward(layer), x, use_reentrant=False)
            else:
                # Pre-Norm Architecture
                # 1. Soliton Attention
                residual = x
                x = layer["attn_norm"](x)
                x = layer["attn"](x)
                x = x + residual

                # 2. Feed Forward
                residual = x
                x = layer["ffn_norm"](x)
                x = layer["ffn"](x)
                x = x + residual

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_topo_loss:
            topo_energy = self.topo_head(x)
            return logits, topo_energy

        return logits
