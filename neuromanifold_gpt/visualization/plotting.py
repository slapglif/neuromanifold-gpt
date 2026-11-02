from typing import Optional

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class WaveFieldVisualizer:
    """
    Visualization tools for wave field evolution and soliton dynamics.
    """

    def __init__(self):
        pass

    def plot_wave_evolution(
        self, wave_field: torch.Tensor, save_path: Optional[str] = None
    ):
        if isinstance(wave_field, torch.Tensor):
            wave_field = wave_field.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(wave_field.T, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xlabel("Time")
        ax.set_ylabel("Space")
        ax.set_title("Wave Field Evolution")
        plt.colorbar(im, ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return fig

    def plot_soliton_collision(
        self, soliton_data: torch.Tensor, save_path: Optional[str] = None
    ):
        if isinstance(soliton_data, torch.Tensor):
            soliton_data = soliton_data.detach().cpu().numpy()

        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(131)
        ax1.plot(soliton_data[0])
        ax1.set_title("Initial State")
        ax1.set_xlabel("Position")

        mid_idx = soliton_data.shape[0] // 2
        ax2 = fig.add_subplot(132)
        ax2.plot(soliton_data[mid_idx])
        ax2.set_title("During Collision")
        ax2.set_xlabel("Position")

        ax3 = fig.add_subplot(133)
        ax3.plot(soliton_data[-1])
        ax3.set_title("Final State")
        ax3.set_xlabel("Position")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return fig

    def plot_braid_trajectory(
        self, trajectories: torch.Tensor, save_path: Optional[str] = None
    ):
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 8))

        num_strands = trajectories.shape[1]
        colors = plt.cm.rainbow(np.linspace(0, 1, num_strands))

        for i in range(num_strands):
            ax.plot(
                trajectories[:, i, 0],
                trajectories[:, i, 1],
                color=colors[i],
                linewidth=2,
                label=f"Strand {i+1}",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Braid Trajectory")
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return fig
