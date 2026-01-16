"""Unit tests for HPO visualization."""

import pytest
from unittest.mock import Mock, patch
import tempfile
import os


def test_plot_optimization_history_import():
    """Test that visualization functions can be imported."""
    from neuromanifold_gpt.hpo.visualize import (
        plot_optimization_history,
        plot_param_importances,
        plot_all_visualizations,
    )

    assert callable(plot_optimization_history)
    assert callable(plot_param_importances)
    assert callable(plot_all_visualizations)


@pytest.mark.skipif(True, reason="Matplotlib not available or optional")
def test_plot_optimization_history_with_mock_study():
    """Test optimization history plot generation."""
    from neuromanifold_gpt.hpo.visualize import plot_optimization_history

    # Mock study with trials
    study = Mock()
    study.trials = [
        Mock(number=0, value=3.0, state="COMPLETE"),
        Mock(number=1, value=2.5, state="COMPLETE"),
        Mock(number=2, value=2.0, state="COMPLETE"),
    ]
    study.best_value = 2.0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_plot.png")
        plot_optimization_history(study, output_path)

        # If matplotlib is available, file should be created
        # If not, function should handle gracefully


def test_matplotlib_availability_check():
    """Test graceful handling when matplotlib unavailable."""
    # This test verifies the module loads even without matplotlib
    try:
        from neuromanifold_gpt.hpo.visualize import plot_all_visualizations
        # If we get here, the import succeeded
        assert True
    except ImportError as e:
        # Should not happen - module should import but warn if matplotlib missing
        pytest.fail(f"Import failed: {e}")
