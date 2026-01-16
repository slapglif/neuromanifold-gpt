"""Tests for config preset discovery utilities."""

import pytest
from pathlib import Path


class TestExtractDescription:
    """Test suite for description extraction from preset files."""

    def test_extract_description_from_nano(self):
        """Test extracting description from nano.py preset."""
        from neuromanifold_gpt.utils.config_presets import _extract_description

        preset_path = Path("neuromanifold_gpt/config/presets/nano.py")
        description = _extract_description(preset_path)

        assert description == "Nano preset for fast experimentation and testing"

    def test_extract_description_from_shakespeare(self):
        """Test extracting description from shakespeare_char.py preset."""
        from neuromanifold_gpt.utils.config_presets import _extract_description

        preset_path = Path("neuromanifold_gpt/config/presets/shakespeare_char.py")
        description = _extract_description(preset_path)

        assert description == "Shakespeare character-level training preset"

    def test_extract_description_nonexistent_file(self):
        """Test description extraction from nonexistent file."""
        from neuromanifold_gpt.utils.config_presets import _extract_description

        preset_path = Path("neuromanifold_gpt/config/presets/nonexistent.py")
        description = _extract_description(preset_path)

        assert description == "Error reading description"


class TestExtractKeySettings:
    """Test suite for key settings extraction from preset files."""

    def test_extract_settings_from_nano(self):
        """Test extracting key settings from nano.py (uses nano config)."""
        from neuromanifold_gpt.utils.config_presets import _extract_key_settings

        preset_path = Path("neuromanifold_gpt/config/presets/nano.py")
        settings = _extract_key_settings(preset_path)

        # Nano preset uses use_nano_config=True, so defaults should be applied
        assert settings['use_nano_config'] is True
        assert settings['n_layer'] == 4  # nano default
        assert settings['n_head'] == 4  # nano default
        assert settings['n_embd'] == 128  # nano default
        assert settings['max_iters'] == 5000

    def test_extract_settings_from_shakespeare(self):
        """Test extracting key settings from shakespeare_char.py (explicit values)."""
        from neuromanifold_gpt.utils.config_presets import _extract_key_settings

        preset_path = Path("neuromanifold_gpt/config/presets/shakespeare_char.py")
        settings = _extract_key_settings(preset_path)

        # Shakespeare has explicit values
        assert settings['use_nano_config'] is False
        assert settings['n_layer'] == 6
        assert settings['n_head'] == 6
        assert settings['n_embd'] == 384
        assert settings['max_iters'] == 5000

    def test_extract_settings_nonexistent_file(self):
        """Test settings extraction from nonexistent file returns None values."""
        from neuromanifold_gpt.utils.config_presets import _extract_key_settings

        preset_path = Path("neuromanifold_gpt/config/presets/nonexistent.py")
        settings = _extract_key_settings(preset_path)

        assert settings['n_layer'] is None
        assert settings['n_head'] is None
        assert settings['n_embd'] is None
        assert settings['max_iters'] is None


class TestEstimateModelSize:
    """Test suite for model size estimation."""

    def test_estimate_nano_size(self):
        """Test model size estimation for nano config (4L, 128E)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_model_size

        # Nano: 12 * 4 * 128^2 = 786,432 params
        size = _estimate_model_size(n_layer=4, n_embd=128)
        assert size == "~786K"

    def test_estimate_shakespeare_size(self):
        """Test model size estimation for shakespeare config (6L, 384E)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_model_size

        # Shakespeare: 12 * 6 * 384^2 = 10,616,832 params
        size = _estimate_model_size(n_layer=6, n_embd=384)
        assert size == "~10M"

    def test_estimate_small_size(self):
        """Test model size estimation for small config (12L, 768E)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_model_size

        # Small: 12 * 12 * 768^2 = 84,934,656 params
        size = _estimate_model_size(n_layer=12, n_embd=768)
        assert size == "~84M"

    def test_estimate_size_with_none_values(self):
        """Test model size estimation with None values."""
        from neuromanifold_gpt.utils.config_presets import _estimate_model_size

        assert _estimate_model_size(None, 384) == "N/A"
        assert _estimate_model_size(6, None) == "N/A"
        assert _estimate_model_size(None, None) == "N/A"

    def test_estimate_large_model_size(self):
        """Test model size estimation for large models (>1B params)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_model_size

        # 12 * 100 * 2048^2 = ~5B params
        size = _estimate_model_size(n_layer=100, n_embd=2048)
        assert "B" in size
        assert size.startswith("~5")


class TestEstimateTrainingTime:
    """Test suite for training time estimation."""

    def test_estimate_time_minutes(self):
        """Test training time estimation for short training (< 1 hour)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_training_time

        # 1800 iterations = 30 minutes @ 1 iter/sec
        time = _estimate_training_time(1800)
        assert time == "~30m"

    def test_estimate_time_hours(self):
        """Test training time estimation for medium training (< 24 hours)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_training_time

        # 5000 iterations = ~1.3 hours @ 1 iter/sec
        time = _estimate_training_time(5000)
        assert time == "~1h"

        # 36000 iterations = 10 hours @ 1 iter/sec
        time = _estimate_training_time(36000)
        assert time == "~10h"

    def test_estimate_time_days(self):
        """Test training time estimation for long training (> 24 hours)."""
        from neuromanifold_gpt.utils.config_presets import _estimate_training_time

        # 100000 iterations = ~27.7 hours = ~1 day @ 1 iter/sec
        time = _estimate_training_time(100000)
        assert time == "~1d"

        # 600000 iterations = ~166 hours = ~6 days @ 1 iter/sec
        time = _estimate_training_time(600000)
        assert time == "~6d"

    def test_estimate_time_with_none(self):
        """Test training time estimation with None value."""
        from neuromanifold_gpt.utils.config_presets import _estimate_training_time

        assert _estimate_training_time(None) == "N/A"


class TestScanPresets:
    """Test suite for preset directory scanning."""

    def test_scan_presets_finds_all_files(self):
        """Test that all preset files are discovered."""
        from neuromanifold_gpt.utils.config_presets import _scan_presets

        presets = _scan_presets("neuromanifold_gpt/config/presets")

        # Should find at least 5 presets: nano, small, medium, reasoning, shakespeare_char
        assert len(presets) >= 5

        preset_names = [p.name for p in presets]
        assert "nano" in preset_names
        assert "small" in preset_names
        assert "medium" in preset_names
        assert "reasoning" in preset_names
        assert "shakespeare_char" in preset_names

    def test_scan_presets_returns_preset_info(self):
        """Test that scan returns PresetInfo objects with correct structure."""
        from neuromanifold_gpt.utils.config_presets import _scan_presets, PresetInfo

        presets = _scan_presets("neuromanifold_gpt/config/presets")

        # Check first preset has all required fields
        preset = presets[0]
        assert isinstance(preset, PresetInfo)
        assert hasattr(preset, 'name')
        assert hasattr(preset, 'description')
        assert hasattr(preset, 'n_layer')
        assert hasattr(preset, 'n_head')
        assert hasattr(preset, 'n_embd')
        assert hasattr(preset, 'max_iters')
        assert hasattr(preset, 'file_path')

    def test_scan_presets_skips_init_file(self):
        """Test that __init__.py is not included in scan results."""
        from neuromanifold_gpt.utils.config_presets import _scan_presets

        presets = _scan_presets("neuromanifold_gpt/config/presets")

        preset_names = [p.name for p in presets]
        assert "__init__" not in preset_names

    def test_scan_presets_nonexistent_directory(self):
        """Test scanning nonexistent directory returns empty list."""
        from neuromanifold_gpt.utils.config_presets import _scan_presets

        presets = _scan_presets("neuromanifold_gpt/config/nonexistent")
        assert presets == []

    def test_scan_presets_sorted_order(self):
        """Test that presets are returned in sorted order."""
        from neuromanifold_gpt.utils.config_presets import _scan_presets

        presets = _scan_presets("neuromanifold_gpt/config/presets")

        preset_names = [p.name for p in presets]
        assert preset_names == sorted(preset_names)


class TestListPresets:
    """Test suite for list_presets function."""

    def test_list_presets_returns_names(self):
        """Test that list_presets returns list of preset names."""
        from neuromanifold_gpt.utils.config_presets import list_presets

        preset_names = list_presets(show_table=False)

        assert isinstance(preset_names, list)
        assert len(preset_names) >= 5
        assert "nano" in preset_names
        assert "shakespeare_char" in preset_names

    def test_list_presets_with_custom_directory(self):
        """Test list_presets with custom directory path."""
        from neuromanifold_gpt.utils.config_presets import list_presets

        preset_names = list_presets(
            directory="neuromanifold_gpt/config/presets",
            show_table=False
        )

        assert len(preset_names) >= 5

    def test_list_presets_nonexistent_directory(self):
        """Test list_presets with nonexistent directory returns empty list."""
        from neuromanifold_gpt.utils.config_presets import list_presets

        preset_names = list_presets(
            directory="neuromanifold_gpt/config/nonexistent",
            show_table=False
        )

        assert preset_names == []

    def test_list_presets_table_display(self):
        """Test that list_presets with show_table=True doesn't crash."""
        from neuromanifold_gpt.utils.config_presets import list_presets

        # Should not raise an exception
        preset_names = list_presets(show_table=True)

        # Still returns names even when showing table
        assert isinstance(preset_names, list)
        assert len(preset_names) >= 5


class TestGetPresetInfo:
    """Test suite for get_preset_info function."""

    def test_get_preset_info_nano(self):
        """Test getting detailed info for nano preset."""
        from neuromanifold_gpt.utils.config_presets import get_preset_info

        info = get_preset_info("nano")

        assert info is not None
        assert info['name'] == "nano"
        assert info['description'] == "Nano preset for fast experimentation and testing"
        assert info['n_layer'] == 4
        assert info['n_head'] == 4
        assert info['n_embd'] == 128
        assert info['max_iters'] == 5000
        assert info['model_size'] == "~786K"
        assert info['training_time'] == "~1h"

    def test_get_preset_info_shakespeare(self):
        """Test getting detailed info for shakespeare preset."""
        from neuromanifold_gpt.utils.config_presets import get_preset_info

        info = get_preset_info("shakespeare_char")

        assert info is not None
        assert info['name'] == "shakespeare_char"
        assert "Shakespeare" in info['description']
        assert info['n_layer'] == 6
        assert info['n_head'] == 6
        assert info['n_embd'] == 384
        assert info['max_iters'] == 5000

    def test_get_preset_info_nonexistent(self):
        """Test getting info for nonexistent preset returns None."""
        from neuromanifold_gpt.utils.config_presets import get_preset_info

        info = get_preset_info("nonexistent_preset")

        assert info is None

    def test_get_preset_info_with_custom_directory(self):
        """Test getting preset info with custom directory path."""
        from neuromanifold_gpt.utils.config_presets import get_preset_info

        info = get_preset_info(
            "nano",
            directory="neuromanifold_gpt/config/presets"
        )

        assert info is not None
        assert info['name'] == "nano"

    def test_get_preset_info_contains_all_fields(self):
        """Test that preset info contains all expected fields."""
        from neuromanifold_gpt.utils.config_presets import get_preset_info

        info = get_preset_info("nano")

        required_fields = [
            'name', 'description', 'file_path',
            'n_layer', 'n_head', 'n_embd', 'max_iters',
            'model_size', 'training_time'
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"


class TestPresetInfoNamedTuple:
    """Test suite for PresetInfo NamedTuple."""

    def test_preset_info_creation(self):
        """Test creating PresetInfo object."""
        from neuromanifold_gpt.utils.config_presets import PresetInfo

        preset = PresetInfo(
            name="test",
            description="Test preset",
            n_layer=4,
            n_head=4,
            n_embd=128,
            max_iters=1000,
            file_path="/path/to/test.py"
        )

        assert preset.name == "test"
        assert preset.description == "Test preset"
        assert preset.n_layer == 4
        assert preset.n_head == 4
        assert preset.n_embd == 128
        assert preset.max_iters == 1000
        assert preset.file_path == "/path/to/test.py"

    def test_preset_info_with_none_values(self):
        """Test creating PresetInfo with None values."""
        from neuromanifold_gpt.utils.config_presets import PresetInfo

        preset = PresetInfo(
            name="test",
            description="Test",
            n_layer=None,
            n_head=None,
            n_embd=None,
            max_iters=None,
            file_path="/path/to/test.py"
        )

        assert preset.n_layer is None
        assert preset.n_head is None
        assert preset.n_embd is None
        assert preset.max_iters is None
