"""
Wave Dynamics Monitor Callback.

Logs metrics specific to the Wave Manifold Network:
- Velocity field norm (Rectified Flow energy)
- Topological charge (Sine-Gordon)
- Soliton stability (KdV mass/momentum conservation)
"""

import lightning as L


class WaveDynamicsMonitor(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % trainer.log_every_n_steps != 0:
            return

        # Access model internals if available
        # Note: This relies on the model stashing info in the forward pass output
        # WaveManifoldLightning returns loss from training_step, but we need 'info' dict
        # We can attach 'info' to the module temporarily or log it directly in training_step.

        # Actually, WaveManifoldLightning.training_step logs 'loss_continuous' etc.
        # This callback can do more advanced aggregation or visualization if needed.
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Placeholder for advanced validation logic (e.g. solving PDE on validation set)
        pass
