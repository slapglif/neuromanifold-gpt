"""Tests for default configuration files."""

import os

import pytest


class TestGPT2SingleGPUConfig:
    """Test suite for GPT-2 124M single GPU configuration."""

    def test_config_loads(self):
        """Test that single GPU config loads without errors."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"

        # Load config using exec
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Verify required variables are present
        assert "batch_size" in config_vars
        assert "block_size" in config_vars
        assert "gradient_accumulation_steps" in config_vars

    def test_batch_size_calculation(self):
        """Test that batch size calculation is correct for single GPU."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Expected: 4 batch * 1024 block * 12 gradaccum * 1 GPU = 49,152 tokens
        total_tokens = (
            config_vars["batch_size"]
            * config_vars["block_size"]
            * config_vars["gradient_accumulation_steps"]
        )
        assert total_tokens == 49152, f"Expected 49,152 tokens, got {total_tokens}"

    def test_config_values_are_sensible(self):
        """Test that single GPU config has sensible hyperparameter values."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Batch size should be positive and reasonable for single GPU
        assert config_vars["batch_size"] > 0
        assert config_vars["batch_size"] <= 16  # Single GPU shouldn't need huge batch

        # Block size should be power of 2 and reasonable
        assert config_vars["block_size"] in [256, 512, 1024, 2048]

        # Gradient accumulation should be reasonable
        assert config_vars["gradient_accumulation_steps"] > 0
        assert config_vars["gradient_accumulation_steps"] <= 64

    def test_eval_intervals_set(self):
        """Test that evaluation intervals are configured."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "eval_interval" in config_vars
        assert "eval_iters" in config_vars
        assert "log_interval" in config_vars

        assert config_vars["eval_interval"] > 0
        assert config_vars["eval_iters"] > 0
        assert config_vars["log_interval"] > 0

    def test_training_duration_set(self):
        """Test that training duration parameters are set."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "max_iters" in config_vars
        assert "lr_decay_iters" in config_vars

        assert config_vars["max_iters"] > 0
        assert config_vars["lr_decay_iters"] > 0

    def test_wandb_configured(self):
        """Test that W&B logging is configured."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "wandb_log" in config_vars
        assert "wandb_project" in config_vars
        assert "wandb_run_name" in config_vars

        assert isinstance(config_vars["wandb_log"], bool)
        assert isinstance(config_vars["wandb_project"], str)
        assert isinstance(config_vars["wandb_run_name"], str)


class TestGPT2MultiGPUConfig:
    """Test suite for GPT-2 124M multi-GPU configuration."""

    def test_config_loads(self):
        """Test that multi-GPU config loads without errors."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"

        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "batch_size" in config_vars
        assert "block_size" in config_vars
        assert "gradient_accumulation_steps" in config_vars

    def test_batch_size_calculation_multi_gpu(self):
        """Test that batch size calculation is correct for 8 GPUs."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Expected: 12 batch * 1024 block * 40 gradaccum = 491,520 tokens
        # Note: gradient_accumulation_steps = 5 * 8 = 40 (per-GPU setting)
        total_tokens = (
            config_vars["batch_size"]
            * config_vars["block_size"]
            * config_vars["gradient_accumulation_steps"]
        )
        assert total_tokens == 491520, f"Expected 491,520 tokens, got {total_tokens}"

    def test_multi_gpu_larger_than_single(self):
        """Test that multi-GPU config has larger effective batch size."""
        single_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        multi_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"

        single_vars = {}
        with open(single_gpu_path, "r") as f:
            exec(f.read(), single_vars)

        multi_vars = {}
        with open(multi_gpu_path, "r") as f:
            exec(f.read(), multi_vars)

        single_tokens = (
            single_vars["batch_size"]
            * single_vars["block_size"]
            * single_vars["gradient_accumulation_steps"]
        )

        multi_tokens = (
            multi_vars["batch_size"]
            * multi_vars["block_size"]
            * multi_vars["gradient_accumulation_steps"]
        )

        assert multi_tokens > single_tokens, "Multi-GPU should have larger batch size"

    def test_gradient_accumulation_reflects_gpu_count(self):
        """Test that gradient accumulation is scaled for multiple GPUs."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # gradient_accumulation_steps should be 5 * 8 = 40
        assert config_vars["gradient_accumulation_steps"] == 40

    def test_multi_gpu_config_values_sensible(self):
        """Test that multi-GPU config has sensible values."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Batch size per GPU should be reasonable
        assert config_vars["batch_size"] > 0
        assert config_vars["batch_size"] <= 32

        # Block size should match single GPU
        assert config_vars["block_size"] == 1024


class TestGPT2MediumFinetuneConfig:
    """Test suite for GPT-2 medium finetuning configuration."""

    def test_config_loads(self):
        """Test that finetune config loads without errors."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"

        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "init_from" in config_vars
        assert "learning_rate" in config_vars

    def test_init_from_pretrained(self):
        """Test that finetune config initializes from pretrained model."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert config_vars["init_from"] == "gpt2-medium"

    def test_finetuning_learning_rate(self):
        """Test that learning rate is appropriate for finetuning."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Finetuning typically uses smaller LR than pretraining
        lr = config_vars["learning_rate"]
        assert lr > 0
        assert lr < 1e-3, "Finetuning LR should be smaller than typical pretraining LR"

    def test_decay_lr_disabled(self):
        """Test that LR decay is disabled for finetuning."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "decay_lr" in config_vars
        assert config_vars["decay_lr"] is False

    def test_batch_size_calculation_finetune(self):
        """Test that finetune batch size calculation is correct."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        # Expected: 2 batch * 16 gradaccum * 1024 tokens = 32,768 tokens/iter
        total_tokens = (
            config_vars["batch_size"]
            * config_vars["gradient_accumulation_steps"]
            * 1024  # block_size not in config, but implied
        )
        assert total_tokens == 32768, f"Expected 32,768 tokens, got {total_tokens}"

    def test_shorter_training_for_finetune(self):
        """Test that finetuning has fewer iterations than pretraining."""
        finetune_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        single_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"

        finetune_vars = {}
        with open(finetune_path, "r") as f:
            exec(f.read(), finetune_vars)

        single_vars = {}
        with open(single_gpu_path, "r") as f:
            exec(f.read(), single_vars)

        assert finetune_vars["max_iters"] < single_vars["max_iters"]

    def test_checkpoint_saving_config(self):
        """Test that checkpoint saving is configured for finetuning."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        assert "always_save_checkpoint" in config_vars
        assert isinstance(config_vars["always_save_checkpoint"], bool)


class TestConfigCompatibility:
    """Test compatibility across different configurations."""

    def test_all_configs_use_same_block_size(self):
        """Test that configs use compatible block sizes."""
        single_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        multi_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"

        single_vars = {}
        with open(single_gpu_path, "r") as f:
            exec(f.read(), single_vars)

        multi_vars = {}
        with open(multi_gpu_path, "r") as f:
            exec(f.read(), multi_vars)

        # Both training configs should use same block size for consistency
        assert single_vars["block_size"] == multi_vars["block_size"]

    def test_weight_decay_consistency(self):
        """Test that weight decay is consistent across training configs."""
        single_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        multi_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"

        single_vars = {}
        with open(single_gpu_path, "r") as f:
            exec(f.read(), single_vars)

        multi_vars = {}
        with open(multi_gpu_path, "r") as f:
            exec(f.read(), multi_vars)

        # Weight decay should be same for both
        assert "weight_decay" in single_vars
        assert "weight_decay" in multi_vars
        assert single_vars["weight_decay"] == multi_vars["weight_decay"]

    def test_eval_config_consistency(self):
        """Test that eval settings are consistent across configs."""
        single_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        multi_gpu_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"

        single_vars = {}
        with open(single_gpu_path, "r") as f:
            exec(f.read(), single_vars)

        multi_vars = {}
        with open(multi_gpu_path, "r") as f:
            exec(f.read(), multi_vars)

        # Eval settings should be consistent
        assert single_vars["eval_interval"] == multi_vars["eval_interval"]
        assert single_vars["eval_iters"] == multi_vars["eval_iters"]
        assert single_vars["log_interval"] == multi_vars["log_interval"]


class TestConfigFileStructure:
    """Test that config files have proper structure and documentation."""

    def test_single_gpu_has_comments(self):
        """Test that single GPU config has helpful comments."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        with open(config_path, "r") as f:
            content = f.read()

        # Should have comments explaining batch size calculation
        assert "#" in content
        assert "batch" in content.lower() or "token" in content.lower()

    def test_multi_gpu_has_launch_instructions(self):
        """Test that multi-GPU config has launch instructions."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"
        with open(config_path, "r") as f:
            content = f.read()

        # Should mention torchrun or distributed training
        assert "torchrun" in content.lower() or "launch" in content.lower()

    def test_finetune_config_has_model_info(self):
        """Test that finetune config documents the model being used."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        with open(config_path, "r") as f:
            content = f.read()

        # Should mention model size or parameters
        assert "345M" in content or "medium" in content


class TestConfigDefaults:
    """Test that all required default parameters are present."""

    def test_single_gpu_has_all_required_params(self):
        """Test that single GPU config has all required training parameters."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_single_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        required_params = [
            "batch_size",
            "block_size",
            "gradient_accumulation_steps",
            "max_iters",
            "eval_interval",
            "log_interval",
        ]

        for param in required_params:
            assert param in config_vars, f"Missing required parameter: {param}"

    def test_multi_gpu_has_all_required_params(self):
        """Test that multi-GPU config has all required training parameters."""
        config_path = "neuromanifold_gpt/config/defaults/gpt2_124m_multi_gpu.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        required_params = [
            "batch_size",
            "block_size",
            "gradient_accumulation_steps",
            "max_iters",
            "eval_interval",
            "log_interval",
        ]

        for param in required_params:
            assert param in config_vars, f"Missing required parameter: {param}"

    def test_finetune_has_all_required_params(self):
        """Test that finetune config has all required parameters."""
        config_path = "neuromanifold_gpt/config/defaults/finetune_gpt2_medium.py"
        config_vars = {}
        with open(config_path, "r") as f:
            exec(f.read(), config_vars)

        required_params = [
            "init_from",
            "batch_size",
            "gradient_accumulation_steps",
            "max_iters",
            "learning_rate",
            "decay_lr",
        ]

        for param in required_params:
            assert param in config_vars, f"Missing required parameter: {param}"
