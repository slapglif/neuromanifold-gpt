import unittest

import torch

from neuromanifold_gpt.model.kan.wave import WaveKANFFN, WaveKANLinear


class TestWaveKAN(unittest.TestCase):
    def setUp(self):
        self.B, self.T, self.D = 2, 10, 32
        self.H = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_wavekan_linear_shapes(self):
        """Test forward pass shapes for WaveKANLinear"""
        model = WaveKANLinear(self.D, self.H).to(self.device)
        x = torch.randn(self.B, self.T, self.D).to(self.device)

        y = model(x)
        self.assertEqual(y.shape, (self.B, self.T, self.H))

    def test_wavekan_ffn_shapes(self):
        """Test forward pass shapes for WaveKANFFN"""
        model = WaveKANFFN(self.D, self.H).to(self.device)
        x = torch.randn(self.B, self.T, self.D).to(self.device)

        y = model(x)
        self.assertEqual(y.shape, (self.B, self.T, self.D))

    def test_wavelet_types(self):
        """Test all supported wavelet types"""
        for wavelet in ["mexican_hat", "morlet", "dog"]:
            model = WaveKANLinear(self.D, self.H, wavelet_type=wavelet).to(self.device)
            x = torch.randn(self.B, self.T, self.D).to(self.device)
            y = model(x)
            self.assertEqual(y.shape, (self.B, self.T, self.H))

            # Check for NaNs
            self.assertFalse(torch.isnan(y).any(), f"NaNs found in {wavelet}")

    def test_backward_pass(self):
        """Test gradient flow"""
        model = WaveKANLinear(self.D, self.H).to(self.device)
        # Proper initialization for leaf tensor requiring grad
        x = torch.randn(self.B, self.T, self.D, device=self.device, requires_grad=True)

        y = model(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.wavelet_weights.grad)
        self.assertIsNotNone(model.scale.grad)
        self.assertIsNotNone(model.translation.grad)

    def test_numerical_stability(self):
        """Test with large/small inputs"""
        model = WaveKANLinear(self.D, self.H).to(self.device)

        # Large inputs
        x_large = torch.randn(self.B, self.T, self.D).to(self.device) * 100
        y_large = model(x_large)
        self.assertFalse(torch.isnan(y_large).any(), "NaNs with large inputs")

        # Small inputs
        x_small = torch.randn(self.B, self.T, self.D).to(self.device) * 1e-5
        y_small = model(x_small)
        self.assertFalse(torch.isnan(y_small).any(), "NaNs with small inputs")

    def test_causality(self):
        """Verify causality is preserved (changing future tokens doesn't affect past)"""
        model = WaveKANLinear(self.D, self.H).to(self.device)
        x = torch.randn(self.B, self.T, self.D).to(self.device)

        # Original output
        y1 = model(x)

        # Modify last token
        x_mod = x.clone()
        x_mod[:, -1, :] = torch.randn(self.B, self.D).to(self.device)
        y2 = model(x_mod)

        # Output for first token should be identical
        # Using a slightly larger tolerance because of potential numerical noise in sums
        self.assertTrue(torch.allclose(y1[:, 0, :], y2[:, 0, :], atol=1e-6))

        # Output for last token should be different
        self.assertFalse(torch.allclose(y1[:, -1, :], y2[:, -1, :]))


if __name__ == "__main__":
    unittest.main()
