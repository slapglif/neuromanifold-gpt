from typing import Tuple

import numpy as np
import torch


class SolitonDataset:
    """
    Synthetic dataset for soliton collision validation.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_length: int = 256,
        spatial_dim: int = 64,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.spatial_dim = spatial_dim

    def generate_sine_gordon_soliton(
        self, amplitude: float = 1.0, velocity: float = 0.5
    ) -> torch.Tensor:
        """
        Generate Sine-Gordon soliton solution.
        phi(x,t) = 4*arctan(exp(gamma*(x - vt)))
        """
        x = torch.linspace(-10, 10, self.spatial_dim)
        t = torch.linspace(0, 10, self.seq_length)

        gamma = 1.0 / np.sqrt(1 - velocity**2)

        phi = torch.zeros(self.seq_length, self.spatial_dim)

        for i, ti in enumerate(t):
            xi = x - velocity * ti
            phi[i] = 4 * torch.arctan(torch.exp(gamma * xi)) * amplitude

        return phi

    def generate_kdv_soliton(
        self, amplitude: float = 1.0, velocity: float = 1.0
    ) -> torch.Tensor:
        """
        Generate KdV soliton solution.
        u(x,t) = A*sech^2(k*(x - ct))
        """
        x = torch.linspace(-20, 20, self.spatial_dim)
        t = torch.linspace(0, 10, self.seq_length)

        k = np.sqrt(amplitude / 12)
        c = velocity

        u = torch.zeros(self.seq_length, self.spatial_dim)

        for i, ti in enumerate(t):
            xi = x - c * ti
            u[i] = amplitude / torch.cosh(k * xi) ** 2

        return u

    def generate_collision(
        self, soliton_type: str = "sine_gordon"
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate two-soliton collision.
        """
        if soliton_type == "sine_gordon":
            soliton1 = self.generate_sine_gordon_soliton(amplitude=1.0, velocity=0.5)
            soliton2 = self.generate_sine_gordon_soliton(amplitude=1.0, velocity=-0.5)
        else:
            soliton1 = self.generate_kdv_soliton(amplitude=1.0, velocity=1.0)
            soliton2 = self.generate_kdv_soliton(amplitude=0.5, velocity=0.5)

        collision = soliton1 + torch.flip(soliton2, [1])

        metadata = {
            "type": soliton_type,
            "elastic": True,
            "num_solitons": 2,
        }

        return collision, metadata

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        collision, metadata = self.generate_collision()
        return collision.unsqueeze(0), metadata
