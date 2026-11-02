"""
PyTorch Lightning Module for WaveManifoldGPT.

Handles:
- Training loop
- Loss aggregation (Discrete + Continuous + Topological)
- Optimization
- Logging
"""


import lightning as L
import torch
import torch.optim as optim

from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.model.wave_manifold_gpt import WaveManifoldGPT


class WaveManifoldLightning(L.LightningModule):
    def __init__(self, config: WaveManifoldConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = WaveManifoldGPT(config)

    def forward(self, idx, targets=None):
        return self.model(idx, targets)

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss, info = self.model(idx, targets)

        # Log all losses
        self.log("train/loss", loss, prog_bar=True)
        if "loss_discrete" in info:
            self.log("train/loss_discrete", info["loss_discrete"])
        if "loss_continuous" in info:
            self.log("train/loss_continuous", info["loss_continuous"])

        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss, info = self.model(idx, targets)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Weight decay handling
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif "latents" in pn:  # specific to some modules
                    no_decay.add(fpn)

        # Validate
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        decay & no_decay
        union_params = decay | no_decay
        # Add remaining (like KAN parameters which might not match standard Linear)
        # KAN spline weights should probably decay
        for pn in param_dict.keys():
            if pn not in union_params:
                # Default to decay for unknown params (like KAN weights)
                decay.add(pn)

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optim.AdamW(
            optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.95)
        )
        return optimizer
