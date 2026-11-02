import pytest
import torch

from neuromanifold_gpt.topology.homology.persistent import PersistentHomology
from neuromanifold_gpt.visualization.plotting import WaveFieldVisualizer


def test_persistent_homology_distance_matrix():
    ph = PersistentHomology()
    embeddings = torch.randn(2, 10, 64)

    dist_matrix = ph.compute_distance_matrix(embeddings)

    assert dist_matrix.shape == (20, 20)


def test_persistent_homology_smoothness():
    ph = PersistentHomology()
    embeddings = torch.randn(2, 10, 64)

    loss = ph.compute_smoothness_loss(embeddings)

    assert loss.item() >= 0


def test_persistent_homology_forward():
    ph = PersistentHomology()
    embeddings = torch.randn(2, 10, 64, requires_grad=True)

    loss = ph(embeddings)
    loss.backward()

    assert embeddings.grad is not None


def test_wave_field_visualizer():
    viz = WaveFieldVisualizer()
    wave_field = torch.randn(100, 64)

    fig = viz.plot_wave_evolution(wave_field)

    assert fig is not None


def test_soliton_collision_viz():
    viz = WaveFieldVisualizer()
    soliton_data = torch.randn(100, 64)

    fig = viz.plot_soliton_collision(soliton_data)

    assert fig is not None


def test_braid_trajectory_viz():
    viz = WaveFieldVisualizer()
    trajectories = torch.randn(50, 5, 2)

    fig = viz.plot_braid_trajectory(trajectories)

    assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
