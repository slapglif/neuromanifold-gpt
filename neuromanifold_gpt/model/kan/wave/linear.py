import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import wavekan_activation


class WaveKANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet_type: str = "mexican_hat",
        bias: bool = False,
        use_base_linear: bool = True,
        use_fast_wavekan: bool = False, # If True, share a/b across outputs (In-wise only)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.use_base_linear = use_base_linear
        self.use_fast_wavekan = use_fast_wavekan

        # Learnable parameters for wavelet transformation
        if self.use_fast_wavekan:
            # Shared shape parameters per input feature
            self.scale = nn.Parameter(torch.ones(in_features))
            self.translation = nn.Parameter(torch.zeros(in_features))
            # Mixing weights are still fully connected
            self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))
        else:
            # Full KAN: unique shape per edge
            self.scale = nn.Parameter(torch.ones(out_features, in_features))
            self.translation = nn.Parameter(torch.zeros(out_features, in_features))
            self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))

        # Initialization
        # Kaiming uniform is good for weights, but scale/trans need care
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))

        # Scale: should be non-zero (1.0 is good)
        # Translation: 0.0 is good
        # But we should allow them to learn easily.
        # Maybe random initialization helps cover the space?
        # Standard KAN uses uniform grid. WaveKAN uses learned a, b.
        # Let's add some noise to translation to cover different parts of input
        nn.init.uniform_(self.translation, -0.5, 0.5)
        # Randomize scale slightly around 1.0 (0.5 to 1.5)
        nn.init.uniform_(self.scale, 0.5, 1.5)

        # Base linear layer (residual connection for stability)
        if self.use_base_linear:
            self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        if bias and not self.use_base_linear:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Input normalization to ensure wavelets operate in active region
        # Use LayerNorm to preserve causality (normalize over features, not time)
        self.layer_norm = nn.LayerNorm(in_features)

    def wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch*Seq, In)

        if self.use_fast_wavekan:
            # Fast Mode: a, b are (In,)
            # We compute psi(x) once per input feature

            # Parametrization: Ensure scale > epsilon
            # We use softplus to enforce positivity + small constant
            scale = F.softplus(self.scale) + 0.1

            # x_scaled: (N, In)
            x_scaled = (x - self.translation.unsqueeze(0)) / scale.unsqueeze(0)

            # Use JIT compiled activation
            wavelet = wavekan_activation(x_scaled, self.wavelet_type)

            # Clamp output to prevent downstream explosions
            wavelet = torch.clamp(wavelet, -10.0, 10.0)

            # Now we have wavelet features: (N, In)
            # We need to mix them: (N, In) @ (Out, In).T -> (N, Out)
            output = F.linear(wavelet, self.wavelet_weights)
            return output

        else:
            # Full Mode: a, b are (Out, In)
            # Expand for broadcasting: (Batch*Seq, Out, In)
            # We want to apply each output neuron's specific scale/translation to the input
            x_expanded = x.unsqueeze(1)  # (N, 1, In)

            # Broadcast parameters: (1, Out, In)
            translation = self.translation.unsqueeze(0)
            scale_param = self.scale.unsqueeze(0)
            scale = F.softplus(scale_param) + 0.1

            # Calculate argument: (x - b) / a
            x_scaled = (x_expanded - translation) / scale

            # Use JIT compiled activation
            wavelet = wavekan_activation(x_scaled, self.wavelet_type)

            # Clamp
            wavelet = torch.clamp(wavelet, -10.0, 10.0)

            # Weight the wavelets: w_ij * phi(x_j)
            weights_expanded = self.wavelet_weights.unsqueeze(0)
            wavelet_weighted = wavelet * weights_expanded

            # Sum over input features: sum_j (w_ij * phi(x_j))
            output = wavelet_weighted.sum(dim=2) # (N, Out)

            return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Input Normalization (Crucial for Wavelets)
        # Use LayerNorm over the last dimension (features) to preserve causality
        x_norm = self.layer_norm(x)

        # Flatten for processing
        x_flat = x_norm.reshape(B * T, D)

        # Wavelet path
        wavelet_out = self.wavelet_transform(x_flat)

        # Base linear path (if enabled)
        if self.use_base_linear:
            # Apply base linear to original x (or normalized? usually original to preserve range info)
            # But KAN usually adds SiLU(x)*w_base.
            # Standard implementation uses Linear(SiLU(x)).
            # Let's use simple Linear(x) as residual, consistent with ChebyKAN's prompt description
            # (though ChebyKAN code I read didn't have it, I'll add it here for robustness)
            base_out = self.base_linear(x.reshape(B * T, D))
            output = wavelet_out + F.silu(base_out) # SiLU activation on base path is common in KAN
        else:
            output = wavelet_out
            if self.bias is not None:
                output = output + self.bias

        return output.reshape(B, T, self.out_features)
