"""Unit tests for Optuna HPO wrapper."""

from unittest.mock import Mock

import pytest

from neuromanifold_gpt.hpo.optuna_search import OptunaHPO


@pytest.fixture
def sample_config():
    """Sample HPO configuration for testing."""
    return {
        "search_space": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
        "fixed_params": {
            "dataset": "shakespeare_char",
            "block_size": 256,
            "max_iters": 100,
            "model_type": "neuromanifold",
        },
        "study": {
            "name": "test-study",
            "direction": "minimize",
            "n_trials": 2,
            "sampler": "random",
        },
    }


def test_optuna_hpo_init(sample_config):
    """Test OptunaHPO initialization."""
    hpo = OptunaHPO(sample_config)

    assert hpo.config == sample_config
    assert hpo.search_space is not None
    assert hpo.study_config["name"] == "test-study"
    assert hpo.study_config["direction"] == "minimize"


def test_validate_study_config_missing_direction():
    """Test validation fails when direction is missing."""
    config = {
        "search_space": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}
        },
        "fixed_params": {},
        "study": {"name": "test"},  # Missing direction
    }

    with pytest.raises(ValueError, match="direction must be specified"):
        OptunaHPO(config)


def test_validate_study_config_invalid_direction():
    """Test validation fails for invalid direction."""
    config = {
        "search_space": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}
        },
        "fixed_params": {},
        "study": {"direction": "invalid"},
    }

    with pytest.raises(ValueError, match="Invalid direction"):
        OptunaHPO(config)


def test_create_study(sample_config):
    """Test Optuna study creation."""
    hpo = OptunaHPO(sample_config)
    study = hpo._create_study()

    assert study is not None
    assert study.study_name == "test-study"
    assert study.direction.name == "MINIMIZE"


def test_create_study_with_different_samplers(sample_config):
    """Test study creation with different sampler types."""
    samplers = [
        "tpe",
        "random",
        "cmaes",
    ]  # Removed "grid" - GridSampler requires search_space parameter

    for sampler_name in samplers:
        config = sample_config.copy()
        config["study"]["sampler"] = sampler_name

        hpo = OptunaHPO(config)
        study = hpo._create_study()

        assert study is not None


def test_create_train_config(sample_config):
    """Test TrainConfig creation from parameters."""
    hpo = OptunaHPO(sample_config)

    params = {
        "learning_rate": 0.001,
        "dataset": "shakespeare_char",
        "block_size": 256,
        "max_iters": 100,
        "model_type": "neuromanifold",
        # Add minimum required params for TrainConfig
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 256,
        "batch_size": 32,
    }

    train_config = hpo._create_train_config(params, trial_number=0)

    assert train_config.learning_rate == 0.001
    assert train_config.max_iters == 100
    assert train_config.save_checkpoints is False  # Should be disabled during HPO
    assert "trial_0000" in train_config.out_dir


def test_export_best_config_format(sample_config, tmp_path):
    """Test best config export creates valid Python file."""
    hpo = OptunaHPO(sample_config)

    # Create a mock study with best trial
    mock_study = Mock()
    mock_study.best_trial = Mock()
    mock_study.best_trial.params = {
        "learning_rate": 0.001,
    }
    mock_study.best_value = 2.5
    mock_study.best_trial.number = 5

    # Set the study on the HPO instance
    hpo.study = mock_study

    # Export to temp file
    output_path = tmp_path / "test_best.py"
    hpo.export_best_config(str(output_path))

    # Verify file exists and is valid Python
    assert output_path.exists()
    content = output_path.read_text()
    assert "learning_rate = 0.001" in content

    # Verify it's importable (valid Python syntax)
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        import test_best

        assert test_best.learning_rate == 0.001
    finally:
        sys.path.remove(str(tmp_path))
