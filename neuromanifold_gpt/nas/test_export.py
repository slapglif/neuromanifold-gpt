"""Unit tests for export functionality.

Tests cover:
- Export to Python config files
- Export to JSON
- Summary report generation
- Directory creation
- File naming
"""

import json
import os
import shutil
import tempfile

import pytest

from neuromanifold_gpt.nas.evaluator import EvaluationResult
from neuromanifold_gpt.nas.export import (
    export_config,
    export_to_json,
    generate_summary_report,
)
from neuromanifold_gpt.nas.search_space import ArchitectureConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    # Cleanup after test
    shutil.rmtree(dirpath)


@pytest.fixture
def sample_architecture():
    """Create a sample architecture for testing."""
    return ArchitectureConfig(
        n_layer=8,
        n_embd=512,
        n_heads=8,
        attention_type="fhn",
        use_kan=True,
        kan_type="faster",
        use_sdr=True,
        sdr_size=2048,
        architecture_id="test_arch_001",
    )


@pytest.fixture
def sample_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        success=True,
        final_loss=2.3,
        perplexity=10.0,
        n_params=15_000_000,
        training_time=120.0,
        tokens_per_second=500.0,
        architecture_id="test_arch_001",
        error_message=None,
    )


class TestExportConfig:
    """Test export_config functionality."""

    def test_export_config_creates_file(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that export_config creates a Python file."""
        output_path = export_config(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
            filename="test_config.py",
        )

        assert os.path.exists(output_path)
        assert output_path.endswith(".py")

    def test_export_config_creates_directory(self, sample_architecture, sample_result):
        """Test that export_config creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "configs", "exported")
            output_path = export_config(
                sample_architecture,
                sample_result,
                output_dir=nested_dir,
            )
            assert os.path.exists(output_path)
            assert os.path.exists(nested_dir)

    def test_export_config_file_is_valid_python(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that exported config is valid Python syntax."""
        output_path = export_config(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        # Try to compile the file
        with open(output_path, "r") as f:
            code = f.read()

        # Should not raise SyntaxError
        compile(code, output_path, "exec")

    def test_export_config_contains_architecture_params(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that exported config contains architecture parameters."""
        output_path = export_config(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            content = f.read()

        # Check for key parameters
        assert str(sample_architecture.n_layer) in content
        assert str(sample_architecture.n_embd) in content
        assert str(sample_architecture.n_heads) in content
        assert sample_architecture.attention_type in content

    def test_export_config_contains_sdr_params(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that exported config contains SDR parameters."""
        output_path = export_config(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            content = f.read()

        # Check for SDR parameters
        assert "use_sdr" in content
        assert "sdr_size" in content

    def test_export_config_contains_performance_metrics(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that exported config contains performance metrics in comments."""
        output_path = export_config(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            content = f.read()

        # Check for performance metrics
        assert str(sample_result.perplexity) in content
        assert str(sample_result.n_params) in content

    def test_export_config_filename_generation(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test automatic filename generation."""
        output_path = export_config(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        filename = os.path.basename(output_path)
        # Should include architecture_id
        assert "test_arch_001" in filename
        assert filename.endswith(".py")


class TestExportToJson:
    """Test export_to_json functionality."""

    def test_export_to_json_creates_file(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that export_to_json creates a JSON file."""
        output_path = export_to_json(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
            filename="test_config.json",
        )

        assert os.path.exists(output_path)
        assert output_path.endswith(".json")

    def test_export_to_json_valid_json(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that exported JSON is valid."""
        output_path = export_to_json(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_export_to_json_contains_architecture(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that JSON contains architecture parameters."""
        output_path = export_to_json(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "architecture" in data
        assert data["architecture"]["n_layer"] == sample_architecture.n_layer
        assert data["architecture"]["n_embd"] == sample_architecture.n_embd
        assert (
            data["architecture"]["attention_type"] == sample_architecture.attention_type
        )

    def test_export_to_json_contains_metrics(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that JSON contains performance metrics."""
        output_path = export_to_json(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "metrics" in data
        assert data["metrics"]["perplexity"] == sample_result.perplexity
        assert data["metrics"]["loss"] == sample_result.final_loss

    def test_export_to_json_with_sdr_params(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that JSON includes SDR parameters."""
        output_path = export_to_json(
            sample_architecture,
            sample_result,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "use_sdr" in data["architecture"]
        assert "sdr_size" in data["architecture"]


class TestGenerateSummaryReport:
    """Test generate_summary_report functionality."""

    def test_generate_summary_creates_file(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that generate_summary_report creates a markdown file."""
        architectures = [sample_architecture]
        results = [sample_result]

        output_path = generate_summary_report(
            architectures, results, output_dir=temp_dir, filename="summary.md"
        )

        assert os.path.exists(output_path)
        assert output_path.endswith(".md")

    def test_generate_summary_contains_metrics(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that summary contains performance metrics."""
        architectures = [sample_architecture]
        results = [sample_result]

        output_path = generate_summary_report(
            architectures,
            results,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            content = f.read()

        # Check for metrics
        assert str(sample_result.perplexity) in content
        assert str(sample_result.n_params) in content

    def test_generate_summary_with_multiple_architectures(self, temp_dir):
        """Test summary with multiple architectures."""
        architectures = [
            ArchitectureConfig(n_layer=6, n_embd=384, architecture_id="arch1"),
            ArchitectureConfig(n_layer=8, n_embd=512, architecture_id="arch2"),
            ArchitectureConfig(n_layer=12, n_embd=768, architecture_id="arch3"),
        ]
        results = [
            EvaluationResult(True, 2.0, 7.39, 10_000_000, 60.0, 500.0, "arch1", None),
            EvaluationResult(True, 1.8, 6.05, 15_000_000, 90.0, 400.0, "arch2", None),
            EvaluationResult(True, 1.6, 4.95, 25_000_000, 150.0, 300.0, "arch3", None),
        ]

        output_path = generate_summary_report(
            architectures,
            results,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            content = f.read()

        # Should contain all architecture IDs
        assert "arch1" in content
        assert "arch2" in content
        assert "arch3" in content

    def test_generate_summary_markdown_format(
        self, temp_dir, sample_architecture, sample_result
    ):
        """Test that summary uses markdown formatting."""
        architectures = [sample_architecture]
        results = [sample_result]

        output_path = generate_summary_report(
            architectures,
            results,
            output_dir=temp_dir,
        )

        with open(output_path, "r") as f:
            content = f.read()

        # Check for markdown elements
        assert "#" in content  # Headers
        assert "|" in content or "-" in content  # Tables or lists


class TestExportEdgeCases:
    """Test edge cases in export functionality."""

    def test_export_config_with_no_result(self, temp_dir):
        """Test export_config when result is None."""
        arch = ArchitectureConfig(architecture_id="test")
        # Should handle None result gracefully
        output_path = export_config(arch, None, output_dir=temp_dir)
        assert os.path.exists(output_path)

    def test_export_with_special_characters_in_id(self, temp_dir):
        """Test export with special characters in architecture_id."""
        arch = ArchitectureConfig(
            architecture_id="test/arch:001",
            n_embd=384,
            n_heads=8,
        )
        result = EvaluationResult(
            True, 2.0, 7.39, 10_000_000, 60.0, 500.0, "test/arch:001", None
        )

        # Should handle special characters (replace or escape)
        output_path = export_config(arch, result, output_dir=temp_dir)
        assert os.path.exists(output_path)

    def test_export_empty_list(self, temp_dir):
        """Test generate_summary_report with empty lists."""
        output_path = generate_summary_report([], [], output_dir=temp_dir)
        assert os.path.exists(output_path)
