"""Unit tests for HPO search space configuration."""

import os
from unittest.mock import Mock

import pytest
import yaml

from neuromanifold_gpt.hpo.search_space import SearchSpace


@pytest.fixture
def sample_config():
    """Sample HPO configuration for testing."""
    return {
        "search_space": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "n_layer": {"type": "int", "low": 2, "high": 8},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
            "use_sdr": {"type": "categorical", "choices": [True, False]},
        },
        "fixed_params": {
            "dataset": "shakespeare_char",
            "block_size": 256,
            "max_iters": 100,
        },
    }


def test_search_space_init(sample_config):
    """Test SearchSpace initialization with valid config."""
    ss = SearchSpace(sample_config)
    assert ss is not None
    assert len(ss.search_params) == 4
    assert len(ss.get_fixed_param_names()) == 3


def test_suggest_float_param_log_scale(sample_config):
    """Test float parameter suggestion with log scale."""
    ss = SearchSpace(sample_config)
    trial = Mock()
    trial.suggest_float = Mock(return_value=0.0001)

    ss.suggest_params(trial)

    # Verify suggest_float called with log=True for learning_rate
    trial.suggest_float.assert_any_call("learning_rate", 1e-5, 1e-2, log=True)


def test_suggest_int_param(sample_config):
    """Test integer parameter suggestion."""
    ss = SearchSpace(sample_config)
    trial = Mock()
    trial.suggest_int = Mock(return_value=4)
    trial.suggest_float = Mock(return_value=0.0001)
    trial.suggest_categorical = Mock(return_value=32)

    params = ss.suggest_params(trial)

    # Verify suggest_int called for n_layer
    trial.suggest_int.assert_called_with("n_layer", 2, 8)
    assert "n_layer" in params


def test_suggest_categorical_param(sample_config):
    """Test categorical parameter suggestion."""
    ss = SearchSpace(sample_config)
    trial = Mock()
    trial.suggest_int = Mock(return_value=4)
    trial.suggest_float = Mock(return_value=0.0001)
    trial.suggest_categorical = Mock(return_value=True)

    ss.suggest_params(trial)

    # Verify suggest_categorical called
    assert trial.suggest_categorical.call_count == 2  # batch_size and use_sdr


def test_fixed_params_included(sample_config):
    """Test that fixed parameters are included in suggested params."""
    ss = SearchSpace(sample_config)
    trial = Mock()
    trial.suggest_int = Mock(return_value=4)
    trial.suggest_float = Mock(return_value=0.0001)
    trial.suggest_categorical = Mock(return_value=32)

    params = ss.suggest_params(trial)

    # Fixed params should be present
    assert params["dataset"] == "shakespeare_char"
    assert params["block_size"] == 256
    assert params["max_iters"] == 100


def test_invalid_config_missing_search_space():
    """Test error handling for invalid config."""
    config = {"fixed_params": {}}  # Missing search_space

    with pytest.raises((KeyError, ValueError)):
        SearchSpace(config)


def test_load_from_yaml_file():
    """Test loading config from actual YAML file."""
    # Test with hpo_config_example.yaml if it exists
    if os.path.exists("hpo_config_example.yaml"):
        with open("hpo_config_example.yaml") as f:
            config = yaml.safe_load(f)

        ss = SearchSpace(config)
        assert ss is not None
        assert len(ss.search_params) > 0


def test_get_fixed_param_names(sample_config):
    """Test retrieval of fixed parameter names."""
    ss = SearchSpace(sample_config)
    fixed_names = ss.get_fixed_param_names()

    assert "dataset" in fixed_names
    assert "block_size" in fixed_names
    assert "max_iters" in fixed_names
    assert "learning_rate" not in fixed_names  # This is a search param


def test_string_representation(sample_config):
    """Test __str__ method."""
    ss = SearchSpace(sample_config)
    str_repr = str(ss)

    assert "SearchSpace" in str_repr
    assert "4" in str_repr  # Number of search params
