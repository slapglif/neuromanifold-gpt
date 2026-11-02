"""Unit tests for random search strategy.

Tests cover:
- SearchResults tracking
- Top-K selection by different metrics
- Checkpoint save/load
- Result accumulation
"""

import json
import os
import shutil
import tempfile

import pytest

from neuromanifold_gpt.nas.evaluator import EvaluationResult
from neuromanifold_gpt.nas.search_space import ArchitectureConfig
from neuromanifold_gpt.nas.strategies.random_search import SearchResults


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def sample_architectures():
    """Create sample architectures for testing."""
    return [
        ArchitectureConfig(n_layer=6, n_embd=384, architecture_id="arch1"),
        ArchitectureConfig(n_layer=8, n_embd=512, architecture_id="arch2"),
        ArchitectureConfig(n_layer=12, n_embd=768, architecture_id="arch3"),
        ArchitectureConfig(n_layer=4, n_embd=256, architecture_id="arch4"),
    ]


@pytest.fixture
def sample_results():
    """Create sample evaluation results."""
    return [
        # arch1: best perplexity, medium params
        EvaluationResult(True, 2.0, 7.39, 10_000_000, 60.0, 500.0, "arch1", None),
        # arch2: medium perplexity, medium params
        EvaluationResult(True, 1.8, 6.05, 15_000_000, 90.0, 400.0, "arch2", None),
        # arch3: lowest perplexity, most params
        EvaluationResult(True, 1.6, 4.95, 25_000_000, 150.0, 300.0, "arch3", None),
        # arch4: highest perplexity, fewest params
        EvaluationResult(True, 2.5, 12.18, 5_000_000, 30.0, 600.0, "arch4", None),
    ]


class TestSearchResultsInitialization:
    """Test SearchResults initialization."""

    def test_empty_initialization(self):
        """Test SearchResults initializes empty."""
        results = SearchResults()
        assert len(results.architectures) == 0
        assert len(results.results) == 0

    def test_initialization_with_data(self, sample_architectures, sample_results):
        """Test SearchResults initializes with data."""
        results = SearchResults(
            architectures=sample_architectures, results=sample_results
        )
        assert len(results.architectures) == 4
        assert len(results.results) == 4


class TestSearchResultsTracking:
    """Test SearchResults tracking functionality."""

    def test_add_result(self, sample_architectures, sample_results):
        """Test adding results incrementally."""
        results = SearchResults()

        assert len(results.architectures) == 0

        results.add_result(sample_architectures[0], sample_results[0])
        assert len(results.architectures) == 1
        assert len(results.results) == 1

        results.add_result(sample_architectures[1], sample_results[1])
        assert len(results.architectures) == 2
        assert len(results.results) == 2

    def test_add_multiple_results(self, sample_architectures, sample_results):
        """Test adding multiple results."""
        results = SearchResults()

        for arch, result in zip(sample_architectures, sample_results):
            results.add_result(arch, result)

        assert len(results.architectures) == 4
        assert len(results.results) == 4


class TestTopKSelection:
    """Test top-K selection functionality."""

    def test_get_top_k_by_perplexity(self, sample_architectures, sample_results):
        """Test selecting top-K architectures by perplexity."""
        results = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        # Get top 2 by perplexity (lower is better)
        top_archs, top_results = results.get_top_k(k=2, metric="perplexity")

        assert len(top_archs) == 2
        assert len(top_results) == 2

        # Should be arch3 (4.95) and arch2 (6.05)
        assert top_archs[0].architecture_id == "arch3"
        assert top_archs[1].architecture_id == "arch2"
        assert top_results[0].perplexity < top_results[1].perplexity

    def test_get_top_k_by_loss(self, sample_architectures, sample_results):
        """Test selecting top-K by loss."""
        results = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        # Get top 3 by loss (lower is better)
        top_archs, top_results = results.get_top_k(k=3, metric="loss")

        assert len(top_archs) == 3

        # Should be sorted by loss (ascending)
        losses = [r.final_loss for r in top_results]
        assert losses == sorted(losses)

    def test_get_top_k_by_params(self, sample_architectures, sample_results):
        """Test selecting top-K by parameter count."""
        results = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        # Get top 2 by params (fewer is better)
        top_archs, top_results = results.get_top_k(k=2, metric="params")

        assert len(top_archs) == 2

        # Should be arch4 (5M) and arch1 (10M)
        assert top_results[0].n_params < top_results[1].n_params

    def test_get_top_k_more_than_available(self, sample_architectures, sample_results):
        """Test requesting more than available results."""
        results = SearchResults(
            architectures=sample_architectures[:2], results=sample_results[:2]
        )

        # Request 5 but only 2 available
        top_archs, top_results = results.get_top_k(k=5, metric="perplexity")

        # Should return all available
        assert len(top_archs) == 2
        assert len(top_results) == 2

    def test_get_top_k_with_failures(self):
        """Test top-K selection with some failed evaluations."""
        architectures = [
            ArchitectureConfig(architecture_id=f"arch{i}") for i in range(5)
        ]
        results_list = [
            EvaluationResult(True, 2.0, 7.39, 10_000_000, 60.0, 500.0, "arch0", None),
            EvaluationResult(
                False, None, None, None, 0.0, None, "arch1", "OOM"
            ),  # Failed
            EvaluationResult(True, 1.8, 6.05, 15_000_000, 90.0, 400.0, "arch2", None),
            EvaluationResult(
                False, None, None, None, 0.0, None, "arch3", "Error"
            ),  # Failed
            EvaluationResult(True, 1.6, 4.95, 25_000_000, 150.0, 300.0, "arch4", None),
        ]

        results = SearchResults(architectures=architectures, results=results_list)

        # Get top 2 - should only include successful ones
        top_archs, top_results = results.get_top_k(k=2, metric="perplexity")

        assert len(top_archs) == 2
        assert all(r.success for r in top_results)


class TestSearchStatistics:
    """Test search statistics generation."""

    def test_get_statistics(self, sample_architectures, sample_results):
        """Test statistics generation."""
        results = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        stats = results.get_statistics()

        assert "total_evaluated" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "best_perplexity" in stats
        assert "best_loss" in stats

        assert stats["total_evaluated"] == 4
        assert stats["successful"] == 4
        assert stats["failed"] == 0
        assert stats["best_perplexity"] == 4.95  # arch3

    def test_get_statistics_with_failures(self):
        """Test statistics with failed evaluations."""
        architectures = [
            ArchitectureConfig(architecture_id=f"arch{i}") for i in range(3)
        ]
        results_list = [
            EvaluationResult(True, 2.0, 7.39, 10_000_000, 60.0, 500.0, "arch0", None),
            EvaluationResult(False, None, None, None, 0.0, None, "arch1", "Error"),
            EvaluationResult(True, 1.8, 6.05, 15_000_000, 90.0, 400.0, "arch2", None),
        ]

        results = SearchResults(architectures=architectures, results=results_list)
        stats = results.get_statistics()

        assert stats["total_evaluated"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1

    def test_get_statistics_empty(self):
        """Test statistics on empty results."""
        results = SearchResults()
        stats = results.get_statistics()

        assert stats["total_evaluated"] == 0
        assert stats["successful"] == 0


class TestCheckpointSaveLoad:
    """Test checkpoint save/load functionality."""

    def test_save_checkpoint(self, temp_dir, sample_architectures, sample_results):
        """Test saving checkpoint to file."""
        results = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        results.save_checkpoint(checkpoint_path)

        assert os.path.exists(checkpoint_path)

    def test_load_checkpoint(self, temp_dir, sample_architectures, sample_results):
        """Test loading checkpoint from file."""
        # Save checkpoint
        results_orig = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        results_orig.save_checkpoint(checkpoint_path)

        # Load checkpoint
        results_loaded = SearchResults.load_checkpoint(checkpoint_path)

        assert len(results_loaded.architectures) == len(results_orig.architectures)
        assert len(results_loaded.results) == len(results_orig.results)

    def test_checkpoint_preserves_data(
        self, temp_dir, sample_architectures, sample_results
    ):
        """Test that save/load preserves all data."""
        results_orig = SearchResults(
            architectures=sample_architectures, results=sample_results
        )

        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        results_orig.save_checkpoint(checkpoint_path)
        results_loaded = SearchResults.load_checkpoint(checkpoint_path)

        # Check architectures
        for orig, loaded in zip(
            results_orig.architectures, results_loaded.architectures
        ):
            assert orig.architecture_id == loaded.architecture_id
            assert orig.n_layer == loaded.n_layer
            assert orig.n_embd == loaded.n_embd

        # Check results
        for orig, loaded in zip(results_orig.results, results_loaded.results):
            assert orig.success == loaded.success
            assert orig.perplexity == loaded.perplexity
            assert orig.n_params == loaded.n_params

    def test_checkpoint_json_format(
        self, temp_dir, sample_architectures, sample_results
    ):
        """Test that checkpoint is valid JSON."""
        results = SearchResults(
            architectures=sample_architectures[:1], results=sample_results[:1]
        )

        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        results.save_checkpoint(checkpoint_path)

        # Verify it's valid JSON
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        assert "architectures" in data
        assert "results" in data
        assert len(data["architectures"]) == 1


class TestSearchResultsEdgeCases:
    """Test edge cases."""

    def test_empty_results(self):
        """Test operations on empty SearchResults."""
        results = SearchResults()

        # get_top_k on empty
        top_archs, top_results = results.get_top_k(k=5)
        assert len(top_archs) == 0

        # get_statistics on empty
        stats = results.get_statistics()
        assert stats["total_evaluated"] == 0

    def test_single_result(self):
        """Test with single result."""
        arch = ArchitectureConfig(architecture_id="single")
        result = EvaluationResult(
            True, 2.0, 7.39, 10_000_000, 60.0, 500.0, "single", None
        )

        results = SearchResults(architectures=[arch], results=[result])

        top_archs, top_results = results.get_top_k(k=1)
        assert len(top_archs) == 1
        assert top_archs[0].architecture_id == "single"

    def test_all_failed_results(self):
        """Test when all evaluations failed."""
        architectures = [
            ArchitectureConfig(architecture_id=f"arch{i}") for i in range(3)
        ]
        results_list = [
            EvaluationResult(False, None, None, None, 0.0, None, f"arch{i}", "Error")
            for i in range(3)
        ]

        results = SearchResults(architectures=architectures, results=results_list)

        # get_top_k should return empty (no successful results)
        top_archs, top_results = results.get_top_k(k=2)
        assert len(top_archs) == 0

        # Statistics should reflect failures
        stats = results.get_statistics()
        assert stats["failed"] == 3
        assert stats["successful"] == 0
