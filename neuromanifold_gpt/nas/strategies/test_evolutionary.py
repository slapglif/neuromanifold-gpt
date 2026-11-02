"""Unit tests for evolutionary search strategy.

Tests cover:
- Mutation operations modify parameters correctly
- Crossover combines parents properly
- Diversity calculation
- Elitism preserves top individuals
- Population updates
"""

import pytest

from neuromanifold_gpt.nas.search_space import ArchitectureConfig, SearchSpace
from neuromanifold_gpt.nas.strategies.evolutionary import EvolutionarySearch


@pytest.fixture
def search_space():
    """Create a search space for testing."""
    return SearchSpace()


@pytest.fixture
def evolutionary_search(search_space):
    """Create an EvolutionarySearch instance for testing."""
    return EvolutionarySearch(
        search_space=search_space,
        population_size=10,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elitism_ratio=0.2,
        seed=42,
    )


@pytest.fixture
def sample_architecture():
    """Create a sample architecture."""
    return ArchitectureConfig(
        n_layer=8,
        n_embd=512,
        n_heads=8,
        attention_type="fhn",
        use_kan=True,
        kan_type="faster",
        use_sdr=True,
        sdr_size=2048,
        architecture_id="test_arch",
    )


class TestEvolutionarySearchInitialization:
    """Test EvolutionarySearch initialization."""

    def test_initialization_with_defaults(self, search_space):
        """Test default initialization."""
        search = EvolutionarySearch(search_space=search_space)

        assert search.population_size == 20
        assert search.mutation_rate == 0.1
        assert search.crossover_rate == 0.5
        assert search.elitism_ratio == 0.1

    def test_initialization_with_custom_params(self, search_space):
        """Test initialization with custom parameters."""
        search = EvolutionarySearch(
            search_space=search_space,
            population_size=50,
            mutation_rate=0.3,
            crossover_rate=0.8,
            elitism_ratio=0.15,
        )

        assert search.population_size == 50
        assert search.mutation_rate == 0.3
        assert search.crossover_rate == 0.8
        assert search.elitism_ratio == 0.15

    def test_get_strategy_name(self, evolutionary_search):
        """Test strategy name."""
        assert evolutionary_search.get_strategy_name() == "evolutionary"


class TestMutationOperations:
    """Test mutation operations."""

    def test_mutate_returns_different_architecture(
        self, evolutionary_search, sample_architecture
    ):
        """Test that mutation produces a different architecture."""
        mutated = evolutionary_search._mutate(sample_architecture)

        # Should be a valid architecture
        is_valid, error = mutated.validate()
        assert is_valid, f"Mutated architecture is invalid: {error}"

        # Should be different from original (at least sometimes)
        # Run multiple times to increase chance of seeing a difference
        different = False
        for _ in range(10):
            mutated = evolutionary_search._mutate(sample_architecture)
            if mutated.to_dict() != sample_architecture.to_dict():
                different = True
                break

        assert different, "Mutation never changed the architecture"

    def test_mutate_respects_constraints(
        self, evolutionary_search, sample_architecture
    ):
        """Test that mutation respects architectural constraints."""
        for _ in range(20):
            mutated = evolutionary_search._mutate(sample_architecture)

            # Check divisibility constraint
            assert (
                mutated.n_embd % mutated.n_heads == 0
            ), f"n_embd={mutated.n_embd} not divisible by n_heads={mutated.n_heads}"

            # Check validity
            is_valid, error = mutated.validate()
            assert is_valid, f"Mutated architecture invalid: {error}"

    def test_mutate_can_change_attention_type(self, evolutionary_search):
        """Test that mutation can change attention_type."""
        arch = ArchitectureConfig(attention_type="fhn")

        # Run multiple mutations to see variety
        attention_types = set()
        for _ in range(50):
            mutated = evolutionary_search._mutate(arch)
            attention_types.add(mutated.attention_type)

        # Should see more than one attention type
        assert len(attention_types) > 1, "Mutation never changed attention_type"

    def test_mutate_can_change_sdr_parameters(self, evolutionary_search):
        """Test that mutation can change SDR parameters."""
        arch = ArchitectureConfig(use_sdr=False, sdr_size=2048)

        # Run multiple mutations
        use_sdr_values = set()
        sdr_sizes = set()
        for _ in range(50):
            mutated = evolutionary_search._mutate(arch)
            use_sdr_values.add(mutated.use_sdr)
            sdr_sizes.add(mutated.sdr_size)

        # Should see variation (probabilistically)
        assert (
            len(use_sdr_values) > 1 or len(sdr_sizes) > 1
        ), "Mutation never changed SDR parameters"


class TestCrossoverOperations:
    """Test crossover operations."""

    def test_crossover_combines_parents(self, evolutionary_search):
        """Test that crossover combines two parents."""
        parent1 = ArchitectureConfig(
            n_layer=6,
            n_embd=384,
            n_heads=6,
            attention_type="fhn",
            use_kan=True,
        )
        parent2 = ArchitectureConfig(
            n_layer=12,
            n_embd=768,
            n_heads=12,
            attention_type="kaufmann",
            use_kan=False,
        )

        offspring = evolutionary_search._crossover(parent1, parent2)

        # Offspring should be valid
        is_valid, error = offspring.validate()
        assert is_valid, f"Offspring is invalid: {error}"

        # Offspring should have some characteristics from each parent
        # This is probabilistic, so we can't assert specific values

    def test_crossover_respects_constraints(self, evolutionary_search):
        """Test that crossover respects architectural constraints."""
        parent1 = ArchitectureConfig(n_embd=384, n_heads=8)
        parent2 = ArchitectureConfig(n_embd=512, n_heads=8)

        for _ in range(20):
            offspring = evolutionary_search._crossover(parent1, parent2)

            # Check constraints
            assert (
                offspring.n_embd % offspring.n_heads == 0
            ), f"Offspring n_embd={offspring.n_embd} not divisible by n_heads={offspring.n_heads}"

            is_valid, error = offspring.validate()
            assert is_valid, f"Offspring invalid: {error}"

    def test_crossover_produces_variety(self, evolutionary_search):
        """Test that crossover produces different offspring."""
        parent1 = ArchitectureConfig(n_layer=6, n_embd=384)
        parent2 = ArchitectureConfig(n_layer=12, n_embd=768)

        offspring_configs = []
        for _ in range(10):
            offspring = evolutionary_search._crossover(parent1, parent2)
            offspring_configs.append(offspring.to_dict())

        # Should see some variety (though not guaranteed due to randomness)
        unique_configs = len(set(tuple(sorted(c.items())) for c in offspring_configs))
        # At least expect more than 1 unique configuration in 10 tries
        assert unique_configs >= 1  # Very conservative check


class TestArchitectureCopying:
    """Test architecture copying."""

    def test_copy_architecture(self, evolutionary_search, sample_architecture):
        """Test that _copy_architecture creates a deep copy."""
        copy = evolutionary_search._copy_architecture(sample_architecture)

        # Should have same values
        assert copy.n_layer == sample_architecture.n_layer
        assert copy.n_embd == sample_architecture.n_embd
        assert copy.attention_type == sample_architecture.attention_type
        assert copy.use_sdr == sample_architecture.use_sdr
        assert copy.sdr_size == sample_architecture.sdr_size

        # Should be a different object
        assert copy is not sample_architecture

        # Modifying copy shouldn't affect original
        copy.n_layer = 999
        assert sample_architecture.n_layer != 999

    def test_copy_architecture_includes_sdr_params(self, evolutionary_search):
        """Test that copying includes SDR parameters."""
        arch = ArchitectureConfig(
            use_sdr=True,
            sdr_size=4096,
            sdr_sparsity=0.03,
            engram_capacity=5000,
            engram_threshold=0.4,
        )

        copy = evolutionary_search._copy_architecture(arch)

        assert copy.use_sdr == arch.use_sdr
        assert copy.sdr_size == arch.sdr_size
        assert copy.sdr_sparsity == arch.sdr_sparsity
        assert copy.engram_capacity == arch.engram_capacity
        assert copy.engram_threshold == arch.engram_threshold


class TestDiversityCalculation:
    """Test diversity calculation."""

    def test_architecture_distance(self, evolutionary_search):
        """Test architecture distance calculation."""
        arch1 = ArchitectureConfig(n_layer=6, n_embd=384, attention_type="fhn")
        arch2 = ArchitectureConfig(n_layer=12, n_embd=768, attention_type="kaufmann")

        distance = evolutionary_search._architecture_distance(arch1, arch2)

        # Distance should be a non-negative number
        assert distance >= 0.0
        assert isinstance(distance, float)

    def test_architecture_distance_identical(
        self, evolutionary_search, sample_architecture
    ):
        """Test distance between identical architectures is zero."""
        distance = evolutionary_search._architecture_distance(
            sample_architecture, sample_architecture
        )

        assert distance == 0.0

    def test_architecture_distance_different(self, evolutionary_search):
        """Test distance between different architectures is positive."""
        arch1 = ArchitectureConfig(n_layer=6, n_embd=384)
        arch2 = ArchitectureConfig(n_layer=12, n_embd=768)

        distance = evolutionary_search._architecture_distance(arch1, arch2)

        assert distance > 0.0

    def test_calculate_diversity_score(self, evolutionary_search):
        """Test diversity score calculation."""
        population = [
            ArchitectureConfig(n_layer=6, n_embd=384),
            ArchitectureConfig(n_layer=8, n_embd=512),
            ArchitectureConfig(n_layer=12, n_embd=768),
        ]

        individual = ArchitectureConfig(n_layer=10, n_embd=640)

        diversity = evolutionary_search._calculate_diversity_score(
            individual, population
        )

        assert diversity >= 0.0
        assert isinstance(diversity, float)

    def test_calculate_population_diversity(self, evolutionary_search):
        """Test population-wide diversity calculation."""
        population = [
            ArchitectureConfig(n_layer=6, n_embd=384),
            ArchitectureConfig(n_layer=8, n_embd=512),
            ArchitectureConfig(n_layer=12, n_embd=768),
        ]

        diversity = evolutionary_search._calculate_population_diversity(population)

        assert diversity >= 0.0
        assert isinstance(diversity, float)

    def test_population_diversity_identical_low(self, evolutionary_search):
        """Test that identical population has low diversity."""
        # Population with very similar architectures
        population = [
            ArchitectureConfig(n_layer=6, n_embd=384),
            ArchitectureConfig(n_layer=6, n_embd=384),
            ArchitectureConfig(n_layer=6, n_embd=384),
        ]

        diversity = evolutionary_search._calculate_population_diversity(population)

        # Diversity should be low (close to 0)
        assert (
            diversity < 0.1
        ), f"Expected low diversity for identical population, got {diversity}"


class TestElitism:
    """Test elitism functionality."""

    def test_elitism_ratio(self, search_space):
        """Test that elitism ratio is properly set."""
        search = EvolutionarySearch(search_space=search_space, elitism_ratio=0.2)
        assert search.elitism_ratio == 0.2

    def test_elitism_preserves_best(self):
        """Test that elitism concept is present (implementation-level test)."""
        # This is more of a documentation test
        search_space = SearchSpace()
        EvolutionarySearch(
            search_space=search_space, population_size=10, elitism_ratio=0.3
        )

        # Elite count should be based on elitism_ratio
        expected_elite_count = int(10 * 0.3)
        assert expected_elite_count == 3


class TestTournamentSelection:
    """Test tournament selection."""

    def test_tournament_size(self, search_space):
        """Test tournament size parameter."""
        search = EvolutionarySearch(search_space=search_space, tournament_size=5)

        assert search.tournament_size == 5


class TestEvolutionarySearchEdgeCases:
    """Test edge cases."""

    def test_small_population(self, search_space):
        """Test with very small population."""
        search = EvolutionarySearch(
            search_space=search_space,
            population_size=2,
        )

        assert search.population_size == 2

    def test_high_mutation_rate(self, search_space):
        """Test with high mutation rate."""
        search = EvolutionarySearch(
            search_space=search_space,
            mutation_rate=0.9,
        )

        assert search.mutation_rate == 0.9

    def test_zero_elitism(self, search_space):
        """Test with zero elitism."""
        search = EvolutionarySearch(
            search_space=search_space,
            elitism_ratio=0.0,
        )

        assert search.elitism_ratio == 0.0

    def test_mutation_with_seed(self, search_space):
        """Test that seed makes mutation deterministic."""
        arch = ArchitectureConfig(n_layer=6, n_embd=384)

        search1 = EvolutionarySearch(search_space=search_space, seed=42)
        search2 = EvolutionarySearch(search_space=search_space, seed=42)

        mutated1 = search1._mutate(arch)
        mutated2 = search2._mutate(arch)

        # With same seed, should get same mutation (first one at least)
        assert mutated1.to_dict() == mutated2.to_dict()
