"""Tests for checkpoint utilities and metadata export."""

import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import pytest


def _import_checkpoints_directly():
    """Import checkpoints module directly without triggering package init.

    This avoids the torch dependency in neuromanifold_gpt/__init__.py,
    allowing tests to run in environments without torch installed.
    """
    checkpoints_path = os.path.join(
        os.path.dirname(__file__), "..", "utils", "checkpoints.py"
    )
    spec = importlib.util.spec_from_file_location(
        "checkpoints_module", checkpoints_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExtractValLoss:
    """Test suite for _extract_val_loss() function."""

    def test_extract_lightning_format(self):
        """Test extracting val_loss from Lightning format filename."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("ckpt-000123-1.2345.ckpt")
        assert val_loss == 1.2345

    def test_extract_lightning_format_different_step(self):
        """Test Lightning format with different step numbers."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("ckpt-999999-0.5678.ckpt")
        assert val_loss == 0.5678

    def test_extract_loss_underscore_format(self):
        """Test extracting val_loss from loss_X.XXXX format."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("checkpoint_loss_2.3456.pt")
        assert val_loss == 2.3456

    def test_extract_loss_hyphen_format(self):
        """Test extracting val_loss from loss-X.XXXX format."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("model-loss-3.1415.pth")
        assert val_loss == 3.1415

    def test_extract_val_underscore_format(self):
        """Test extracting val_loss from val_X.XXXX format."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("checkpoint_val_1.9876.pt")
        assert val_loss == 1.9876

    def test_extract_val_hyphen_format(self):
        """Test extracting val_loss from val-X.XXXX format."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("model-val-0.1234.ckpt")
        assert val_loss == 0.1234

    def test_extract_case_insensitive(self):
        """Test that pattern matching is case-insensitive."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("checkpoint_LOSS_1.5000.pt")
        assert val_loss == 1.5000

        val_loss = checkpoints._extract_val_loss("checkpoint_VAL_2.5000.pt")
        assert val_loss == 2.5000

    def test_extract_no_match_returns_none(self):
        """Test that unrecognized filename returns None."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("checkpoint_epoch_10.pt")
        assert val_loss is None

    def test_extract_no_decimal_returns_none(self):
        """Test that filename without decimal value returns None."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("checkpoint_loss_123.pt")
        assert val_loss is None

    def test_extract_empty_filename(self):
        """Test that empty filename returns None."""
        checkpoints = _import_checkpoints_directly()

        val_loss = checkpoints._extract_val_loss("")
        assert val_loss is None


class TestExtractTrainingStep:
    """Test suite for _extract_training_step() function."""

    def test_extract_lightning_format(self):
        """Test extracting training step from Lightning format."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("ckpt-000123-1.2345.ckpt")
        assert step == 123

    def test_extract_lightning_format_large_step(self):
        """Test Lightning format with large step number."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("ckpt-999999-0.5678.ckpt")
        assert step == 999999

    def test_extract_step_underscore_format(self):
        """Test extracting step from step_XXXXX format."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("checkpoint_step_12345.pt")
        assert step == 12345

    def test_extract_step_hyphen_format(self):
        """Test extracting step from step-XXXXX format."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("model-step-54321.pth")
        assert step == 54321

    def test_extract_iter_underscore_format(self):
        """Test extracting step from iter_XXXXX format."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("checkpoint_iter_67890.pt")
        assert step == 67890

    def test_extract_iter_hyphen_format(self):
        """Test extracting step from iter-XXXXX format."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("model-iter-11111.ckpt")
        assert step == 11111

    def test_extract_case_insensitive(self):
        """Test that pattern matching is case-insensitive."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("checkpoint_STEP_1000.pt")
        assert step == 1000

        step = checkpoints._extract_training_step("checkpoint_ITER_2000.pt")
        assert step == 2000

    def test_extract_no_match_returns_none(self):
        """Test that unrecognized filename returns None."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("checkpoint_epoch_10.pt")
        assert step is None

    def test_extract_empty_filename(self):
        """Test that empty filename returns None."""
        checkpoints = _import_checkpoints_directly()

        step = checkpoints._extract_training_step("")
        assert step is None


class TestFormatAge:
    """Test suite for _format_age() function."""

    def test_format_seconds_ago(self):
        """Test formatting age in seconds."""
        checkpoints = _import_checkpoints_directly()

        # 30 seconds ago
        timestamp = datetime.now().timestamp() - 30
        age_str = checkpoints._format_age(timestamp)
        assert age_str == "30s ago"

    def test_format_minutes_ago(self):
        """Test formatting age in minutes."""
        checkpoints = _import_checkpoints_directly()

        # 5 minutes ago (300 seconds)
        timestamp = datetime.now().timestamp() - 300
        age_str = checkpoints._format_age(timestamp)
        assert age_str == "5m ago"

    def test_format_hours_ago(self):
        """Test formatting age in hours."""
        checkpoints = _import_checkpoints_directly()

        # 3 hours ago (10800 seconds)
        timestamp = datetime.now().timestamp() - 10800
        age_str = checkpoints._format_age(timestamp)
        assert age_str == "3h ago"

    def test_format_days_ago(self):
        """Test formatting age in days."""
        checkpoints = _import_checkpoints_directly()

        # 2 days ago (172800 seconds)
        timestamp = datetime.now().timestamp() - 172800
        age_str = checkpoints._format_age(timestamp)
        assert age_str == "2d ago"

    def test_format_zero_seconds(self):
        """Test formatting age of zero seconds."""
        checkpoints = _import_checkpoints_directly()

        timestamp = datetime.now().timestamp()
        age_str = checkpoints._format_age(timestamp)
        assert age_str == "0s ago"


class TestFormatSize:
    """Test suite for _format_size() function."""

    def test_format_bytes(self):
        """Test formatting size in bytes."""
        checkpoints = _import_checkpoints_directly()

        size_str = checkpoints._format_size(512)
        assert size_str == "512B"

    def test_format_kilobytes(self):
        """Test formatting size in kilobytes."""
        checkpoints = _import_checkpoints_directly()

        size_str = checkpoints._format_size(2048)  # 2KB
        assert size_str == "2.0KB"

    def test_format_megabytes(self):
        """Test formatting size in megabytes."""
        checkpoints = _import_checkpoints_directly()

        size_str = checkpoints._format_size(5 * 1024 * 1024)  # 5MB
        assert size_str == "5.0MB"

    def test_format_gigabytes(self):
        """Test formatting size in gigabytes."""
        checkpoints = _import_checkpoints_directly()

        size_str = checkpoints._format_size(2 * 1024 * 1024 * 1024)  # 2GB
        assert size_str == "2.00GB"

    def test_format_zero_bytes(self):
        """Test formatting zero bytes."""
        checkpoints = _import_checkpoints_directly()

        size_str = checkpoints._format_size(0)
        assert size_str == "0B"

    def test_format_fractional_kb(self):
        """Test formatting fractional kilobytes."""
        checkpoints = _import_checkpoints_directly()

        size_str = checkpoints._format_size(1536)  # 1.5KB
        assert size_str == "1.5KB"


class TestScanCheckpoints:
    """Test suite for _scan_checkpoints() function."""

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning empty directory returns empty list."""
        checkpoints_module = _import_checkpoints_directly()

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert checkpoints == []

    def test_scan_nonexistent_directory(self):
        """Test scanning non-existent directory returns empty list."""
        checkpoints_module = _import_checkpoints_directly()

        checkpoints = checkpoints_module._scan_checkpoints("/nonexistent/path")
        assert checkpoints == []

    def test_scan_finds_pt_files(self, tmp_path):
        """Test scanning finds .pt checkpoint files."""
        checkpoints_module = _import_checkpoints_directly()

        # Create test checkpoint file
        ckpt_file = tmp_path / "checkpoint_loss_1.2345.pt"
        ckpt_file.write_text("dummy checkpoint data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0][0] == "checkpoint_loss_1.2345.pt"
        assert checkpoints[0][1] == 1.2345  # val_loss

    def test_scan_finds_ckpt_files(self, tmp_path):
        """Test scanning finds .ckpt checkpoint files."""
        checkpoints_module = _import_checkpoints_directly()

        # Create test checkpoint file
        ckpt_file = tmp_path / "ckpt-000123-2.3456.ckpt"
        ckpt_file.write_text("dummy checkpoint data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0][0] == "ckpt-000123-2.3456.ckpt"
        assert checkpoints[0][1] == 2.3456  # val_loss

    def test_scan_finds_pth_files(self, tmp_path):
        """Test scanning finds .pth checkpoint files."""
        checkpoints_module = _import_checkpoints_directly()

        # Create test checkpoint file
        ckpt_file = tmp_path / "model-val-3.4567.pth"
        ckpt_file.write_text("dummy checkpoint data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0][0] == "model-val-3.4567.pth"
        assert checkpoints[0][1] == 3.4567  # val_loss

    def test_scan_finds_multiple_files(self, tmp_path):
        """Test scanning finds multiple checkpoint files."""
        checkpoints_module = _import_checkpoints_directly()

        # Create multiple test checkpoint files
        (tmp_path / "checkpoint_1.pt").write_text("data")
        (tmp_path / "checkpoint_2.ckpt").write_text("data")
        (tmp_path / "checkpoint_3.pth").write_text("data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 3

    def test_scan_ignores_non_checkpoint_files(self, tmp_path):
        """Test scanning ignores non-checkpoint files."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint and non-checkpoint files
        (tmp_path / "checkpoint.pt").write_text("data")
        (tmp_path / "readme.txt").write_text("data")
        (tmp_path / "config.json").write_text("data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0][0] == "checkpoint.pt"

    def test_scan_ignores_directories(self, tmp_path):
        """Test scanning ignores directories."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint file and directory
        (tmp_path / "checkpoint.pt").write_text("data")
        (tmp_path / "subdir.pt").mkdir()  # Directory with .pt extension

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0][0] == "checkpoint.pt"

    def test_scan_extracts_file_metadata(self, tmp_path):
        """Test scanning extracts file metadata correctly."""
        checkpoints_module = _import_checkpoints_directly()

        # Create test checkpoint file
        ckpt_file = tmp_path / "checkpoint.pt"
        ckpt_file.write_text("test data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1

        filename, val_loss, mtime, size = checkpoints[0]
        assert filename == "checkpoint.pt"
        assert isinstance(mtime, float)
        assert mtime > 0
        assert size == len("test data")

    def test_scan_handles_no_val_loss(self, tmp_path):
        """Test scanning handles checkpoints without val_loss."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint without val_loss in name
        (tmp_path / "checkpoint_epoch_10.pt").write_text("data")

        checkpoints = checkpoints_module._scan_checkpoints(str(tmp_path))
        assert len(checkpoints) == 1
        assert checkpoints[0][0] == "checkpoint_epoch_10.pt"
        assert checkpoints[0][1] is None  # No val_loss


class TestExportCheckpointsMetadata:
    """Test suite for export_checkpoints_metadata() function."""

    def test_export_creates_json_file(self, tmp_path):
        """Test that export creates a JSON file."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory with test file
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_loss_1.2345.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        assert output_file.exists()

    def test_export_creates_valid_json(self, tmp_path):
        """Test that export creates valid JSON."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory with test file
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_loss_1.2345.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        # Should be able to parse JSON
        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_export_includes_metadata_header(self, tmp_path):
        """Test that export includes metadata header."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "exported_at" in data["metadata"]
        assert "directory" in data["metadata"]
        assert "checkpoint_count" in data["metadata"]

    def test_export_includes_checkpoints_list(self, tmp_path):
        """Test that export includes checkpoints list."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        assert "checkpoints" in data
        assert isinstance(data["checkpoints"], list)

    def test_export_checkpoint_count_matches(self, tmp_path):
        """Test that checkpoint_count matches actual number of checkpoints."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory with multiple files
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_1.pt").write_text("data")
        (ckpt_dir / "checkpoint_2.pt").write_text("data")
        (ckpt_dir / "checkpoint_3.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        assert data["metadata"]["checkpoint_count"] == 3
        assert len(data["checkpoints"]) == 3

    def test_export_checkpoint_fields(self, tmp_path):
        """Test that checkpoint entries have all required fields."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "ckpt-000123-1.2345.ckpt").write_text("test data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        checkpoint = data["checkpoints"][0]
        assert "filename" in checkpoint
        assert "val_loss" in checkpoint
        assert "training_step" in checkpoint
        assert "timestamp" in checkpoint
        assert "age" in checkpoint
        assert "size_bytes" in checkpoint
        assert "size_mb" in checkpoint

    def test_export_extracts_val_loss(self, tmp_path):
        """Test that export correctly extracts val_loss."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint with val_loss in name
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_loss_1.2345.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        checkpoint = data["checkpoints"][0]
        assert checkpoint["val_loss"] == 1.2345

    def test_export_extracts_training_step(self, tmp_path):
        """Test that export correctly extracts training step."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint with training step in name
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "ckpt-000123-1.2345.ckpt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        checkpoint = data["checkpoints"][0]
        assert checkpoint["training_step"] == 123

    def test_export_handles_missing_val_loss(self, tmp_path):
        """Test that export handles checkpoints without val_loss."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint without val_loss
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_epoch_10.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        checkpoint = data["checkpoints"][0]
        assert checkpoint["val_loss"] is None

    def test_export_handles_missing_training_step(self, tmp_path):
        """Test that export handles checkpoints without training step."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint without training step
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint_epoch_10.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        checkpoint = data["checkpoints"][0]
        assert checkpoint["training_step"] is None

    def test_export_calculates_size_mb(self, tmp_path):
        """Test that export correctly calculates size in MB."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint with known size
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        data_content = "x" * 1000000  # 1MB of data
        (ckpt_dir / "checkpoint.pt").write_text(data_content)

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        checkpoint = data["checkpoints"][0]
        assert checkpoint["size_bytes"] == 1000000
        assert checkpoint["size_mb"] == 1.0

    def test_export_empty_directory(self, tmp_path):
        """Test exporting from empty directory."""
        checkpoints_module = _import_checkpoints_directly()

        # Create empty checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        assert data["metadata"]["checkpoint_count"] == 0
        assert data["checkpoints"] == []

    def test_export_nonexistent_directory_raises_error(self, tmp_path):
        """Test that export raises ValueError for non-existent directory."""
        checkpoints_module = _import_checkpoints_directly()

        output_file = tmp_path / "metadata.json"

        with pytest.raises(ValueError, match="Directory does not exist"):
            checkpoints_module.export_checkpoints_metadata(
                "/nonexistent/path", str(output_file)
            )

    def test_export_file_path_raises_error(self, tmp_path):
        """Test that export raises ValueError when directory is a file."""
        checkpoints_module = _import_checkpoints_directly()

        # Create a file instead of directory
        not_a_dir = tmp_path / "not_a_directory.txt"
        not_a_dir.write_text("data")

        output_file = tmp_path / "metadata.json"

        with pytest.raises(ValueError, match="Path is not a directory"):
            checkpoints_module.export_checkpoints_metadata(
                str(not_a_dir), str(output_file)
            )

    def test_export_creates_output_directory(self, tmp_path):
        """Test that export creates output directory if it doesn't exist."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint.pt").write_text("data")

        # Output file in non-existent subdirectory
        output_file = tmp_path / "subdir" / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_export_json_formatting(self, tmp_path):
        """Test that exported JSON is formatted with indentation."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        # Read raw JSON text
        with open(output_file) as f:
            json_text = f.read()

        # Should have newlines (formatted)
        assert "\n" in json_text
        # Should have indentation
        assert "  " in json_text

    def test_export_exported_at_is_iso_format(self, tmp_path):
        """Test that exported_at timestamp is in ISO format."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint directory
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "checkpoint.pt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        # Should be able to parse as ISO timestamp
        exported_at = data["metadata"]["exported_at"]
        datetime.fromisoformat(exported_at)  # Should not raise

    def test_export_multiple_checkpoints_ordering(self, tmp_path):
        """Test that multiple checkpoints are exported."""
        checkpoints_module = _import_checkpoints_directly()

        # Create multiple checkpoints
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "ckpt-001-2.0000.ckpt").write_text("data")
        (ckpt_dir / "ckpt-002-1.5000.ckpt").write_text("data")
        (ckpt_dir / "ckpt-003-1.0000.ckpt").write_text("data")

        output_file = tmp_path / "metadata.json"
        checkpoints_module.export_checkpoints_metadata(str(ckpt_dir), str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        # Should have all 3 checkpoints
        assert len(data["checkpoints"]) == 3

        # Verify all have expected fields
        for checkpoint in data["checkpoints"]:
            assert checkpoint["val_loss"] is not None
            assert checkpoint["training_step"] is not None


class TestListCheckpoints:
    """Test suite for checkpoints_module.list_checkpoints() function."""

    def test_list_empty_directory(self, tmp_path):
        """Test listing checkpoints in empty directory."""
        checkpoints_module = _import_checkpoints_directly()

        filenames = checkpoints_module.list_checkpoints(str(tmp_path))
        assert filenames == []

    def test_list_returns_filenames_only(self, tmp_path):
        """Test that list_checkpoints returns filenames, not full paths."""
        checkpoints_module = _import_checkpoints_directly()

        # Create checkpoint files
        (tmp_path / "checkpoint_1.pt").write_text("data")
        (tmp_path / "checkpoint_2.pt").write_text("data")

        filenames = checkpoints_module.list_checkpoints(str(tmp_path))

        assert len(filenames) == 2
        assert "checkpoint_1.pt" in filenames
        assert "checkpoint_2.pt" in filenames
        # Should not contain full paths
        for filename in filenames:
            assert "/" not in filename and "\\" not in filename

    def test_list_multiple_checkpoints(self, tmp_path):
        """Test listing multiple checkpoint files."""
        checkpoints_module = _import_checkpoints_directly()

        # Create multiple checkpoint files
        (tmp_path / "model_1.pt").write_text("data")
        (tmp_path / "model_2.ckpt").write_text("data")
        (tmp_path / "model_3.pth").write_text("data")

        filenames = checkpoints_module.list_checkpoints(str(tmp_path))
        assert len(filenames) == 3
