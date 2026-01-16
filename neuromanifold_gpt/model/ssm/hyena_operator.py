"""
Hyena Operator: Implicit Convolutional Filter.

Adapted from HyenaDNA (https://github.com/HazyResearch/hyena-dna).
Implements the Hyena hierarchy for sub-quadratic long-range dependency modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

# Optional: Replace torchvision dependency if needed, but we installed it.
# Fallback is necessary for environments where torchvision might be missing or minimal.
try:
    from torchvision.ops import StochasticDepth
except ImportError:
    # Fallback if torchvision is missing
    class StochasticDepth(nn.Module):
        def __init__(self, p: float, mode: str = "row"):
            super().__init__()
            self.p = p
            self.mode = mode

        def forward(self, x):
            if not self.training or self.p == 0.0:
                return x
            if self.mode == "row":
                shape = [x.shape[0]] + [1] * (x.ndim - 1)
            else:
                shape = [1] * x.ndim
            mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(1 - self.p)
            return x * mask / (1 - self.p)


def fftconv(u, k, D):
    """
    Apply convolution via FFT.

    Computes convolution in O(N log N) using the Fast Fourier Transform.
    """
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    """Sin activation for Hyena Filter."""

    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        modulate: bool = True,
        shift: float = 0.05,
        **kwargs,
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,
        order=16,
        fused_fft_conv=False,
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,
        wd=0,
        bias=True,
        num_inner_mlps=2,
        normalized=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and >= 3"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))
        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized

        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None:
            k = self.filter(L)
        k = k[0] if type(k) is tuple else k
        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model,
        l_max,
        order=2,
        filter_order=64,
        dropout=0.0,
        filter_dropout=0.0,
        **filter_args,
    ):
        """
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf
        """
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(inner_width, inner_width, 3, padding=2, groups=inner_width)
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args,
        )

    def forward(self, u, *args, **kwargs):
        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        u = rearrange(u, "b l d -> b d l")

        uc = self.short_filter(u)[..., :l_filter]
        *x, v = uc.split(self.d_model, dim=1)

        k = self.filter_fn.filter(l_filter)[0]
        k = rearrange(k, "l (o d) -> o d l", o=self.order - 1)
        bias = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o])

        y = rearrange(v * x[0], "b d l -> b l d")
        y = self.out_proj(y)
        return y
