
import torch
import torch.nn as nn
from neuromanifold_gpt.model.attention.fhn import FHNAttention
from neuromanifold_gpt.model.attention.knot import KnotAttention
from neuromanifold_gpt.model.embeddings.ramanujan import RamanujanPositionalEmbedding

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
        
        # 1. Topology (Diffusion Matrix)
        # Computes 'Linking Number' matrix -> Attention Weights
        self.knot_gate = KnotAttention(embed_dim, manifold_dim=manifold_dim, n_heads=n_heads)
        
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
        
    def forward(self, x, spectral_basis, coords):
        """
        Args:
            x: (B, T, D) Input features
            spectral_basis: Unused in V2 (We use Knot graph directly)
            coords: (B, T, M) Manifold coordinates for Knot calculation
        """
        B, T, D = x.shape
        H = self.n_heads
        
        # 1. Compute Topology (Knot Attention)
        diffused_signal, knot_info = self.knot_gate(x, coords)
        
        # Normalize input to prevent shock
        # FHN likes inputs in [-1, 1] approx
        I_base = torch.tanh(x) 
        
        # 2. Initialize State (u, w)
        u = I_base
        w = torch.zeros_like(u)
        
        # 3. Reaction-Diffusion Loop
        # Iterate FHN steps, injecting Diffusion at each step
        
        # Clamp dt to safe range [0.01, 0.2]
        dt = torch.sigmoid(self.dt) * 0.2
        
        # Input scaling (learned)
        # We allow the model to scale the input down if it's too strong
        
        for _ in range(self.n_fhn_steps):
            # A. Reaction (FHN Local Dynamics)
            # u is already safe from previous step clamping/norm
            
            # du = u - u^3/3 - w
            # Safe cubic: u^3 can explode if u > 2.
            # We use softsign or tanh for the non-linearity if cubic is too unstable?
            # Stick to cubic but ensure u stays small.
            
            # LayerNorm state to keep it in range?
            # Hard clamping is safer for ODE.
            u_safe = u.clamp(-2.0, 2.0)
            
            du = u_safe - (u_safe**3)/3.0 - w
            
            # Add External Input (Stimulus)
            # I = I_base + diffusion
            # Diffusion is also potentially large. Tanh it.
            I_diff = torch.tanh(diffused_signal)
            
            # Total Drive
            drive = I_base + I_diff
            
            du = du + drive
            
            # Update u
            u_new = u + dt * du
            
            # Update w (Recovery)
            # dw = (u + a - b*w) / tau
            # tau is 12.5.
            dw = (u_safe + self.a - self.b * w) / self.tau
            w_new = w + dt * dw
            
            # Final Safety Clamp
            u = u_new.clamp(-3.0, 3.0)
            w = w_new.clamp(-3.0, 3.0)
            
        # 4. Output
        out = self.out_proj(u)
        
        return out, knot_info
