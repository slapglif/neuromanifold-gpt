import pytest
import torch

from neuromanifold_gpt.training.curriculum import (
    CurriculumConfig,
    CurriculumLoss,
    CurriculumScheduler,
    GenerationMode,
    HybridOutput,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCurriculumScheduler:
    def test_initialization(self):
        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)

        assert scheduler.step == 0
        assert scheduler.get_stage() == GenerationMode.DISCRETE

    def test_discrete_stage(self):
        config = CurriculumConfig(discrete_steps=1000)
        scheduler = CurriculumScheduler(config)

        assert scheduler.get_stage() == GenerationMode.DISCRETE
        assert scheduler.get_alpha() == 0.0

        for _ in range(999):
            scheduler.advance()

        assert scheduler.get_stage() == GenerationMode.DISCRETE

    def test_hybrid_stage(self):
        config = CurriculumConfig(discrete_steps=1000, hybrid_steps=1000)
        scheduler = CurriculumScheduler(config)

        for _ in range(1001):
            scheduler.advance()

        assert scheduler.get_stage() == GenerationMode.HYBRID
        assert 0.0 < scheduler.get_alpha() < 1.0

    def test_continuous_stage(self):
        config = CurriculumConfig(discrete_steps=100, hybrid_steps=100)
        scheduler = CurriculumScheduler(config)

        for _ in range(250):
            scheduler.advance()

        assert scheduler.get_stage() == GenerationMode.CONTINUOUS
        assert scheduler.get_alpha() == 1.0

    def test_alpha_progression(self):
        config = CurriculumConfig(
            discrete_steps=100,
            hybrid_steps=100,
            hybrid_alpha_start=0.0,
            hybrid_alpha_end=1.0,
        )
        scheduler = CurriculumScheduler(config)

        for _ in range(100):
            scheduler.advance()

        alpha_start = scheduler.get_alpha()
        assert alpha_start == pytest.approx(0.0, abs=0.01)

        for _ in range(50):
            scheduler.advance()

        alpha_mid = scheduler.get_alpha()
        assert 0.4 < alpha_mid < 0.6

        for _ in range(50):
            scheduler.advance()

        alpha_end = scheduler.get_alpha()
        assert alpha_end == pytest.approx(1.0, abs=0.01)

    def test_state_dict(self):
        config = CurriculumConfig()
        scheduler1 = CurriculumScheduler(config)

        for _ in range(500):
            scheduler1.advance()

        state = scheduler1.state_dict()

        scheduler2 = CurriculumScheduler(config)
        scheduler2.load_state_dict(state)

        assert scheduler2.step == scheduler1.step
        assert scheduler2.get_stage() == scheduler1.get_stage()


class TestHybridOutput:
    def test_discrete_mode(self, device):
        output_layer = HybridOutput(embed_dim=256, vocab_size=1000).to(device)

        hidden_states = torch.randn(2, 32, 256, device=device)

        outputs = output_layer(
            hidden_states, mode=GenerationMode.DISCRETE, alpha=0.0, temperature=1.0
        )

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 32, 1000)
        assert outputs["mode"] == "discrete"

    def test_continuous_mode(self, device):
        output_layer = HybridOutput(embed_dim=256, vocab_size=1000).to(device)

        hidden_states = torch.randn(2, 32, 256, device=device)

        outputs = output_layer(
            hidden_states, mode=GenerationMode.CONTINUOUS, alpha=1.0, temperature=1.0
        )

        assert "embeddings" in outputs
        assert outputs["embeddings"].shape == (2, 32, 256)
        assert outputs["mode"] == "continuous"

    def test_hybrid_mode(self, device):
        output_layer = HybridOutput(embed_dim=256, vocab_size=1000).to(device)

        hidden_states = torch.randn(2, 32, 256, device=device)

        outputs = output_layer(
            hidden_states, mode=GenerationMode.HYBRID, alpha=0.5, temperature=1.0
        )

        assert "logits" in outputs
        assert "embeddings" in outputs
        assert "discrete_samples" in outputs
        assert outputs["mode"] == "hybrid"
        assert outputs["alpha"] == 0.5

    def test_temperature_scaling(self, device):
        output_layer = HybridOutput(embed_dim=256, vocab_size=1000).to(device)

        hidden_states = torch.randn(2, 32, 256, device=device)

        outputs_temp1 = output_layer(
            hidden_states, GenerationMode.DISCRETE, temperature=1.0
        )
        outputs_temp2 = output_layer(
            hidden_states, GenerationMode.DISCRETE, temperature=2.0
        )

        assert not torch.allclose(outputs_temp1["logits"], outputs_temp2["logits"])


class TestCurriculumLoss:
    def test_discrete_loss(self, device):
        loss_fn = CurriculumLoss().to(device)

        logits = torch.randn(2, 32, 1000, device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)

        outputs = {"logits": logits, "mode": "discrete"}

        loss = loss_fn(outputs, targets, GenerationMode.DISCRETE, alpha=0.0)

        assert loss.item() > 0
        assert loss.shape == ()

    def test_continuous_loss(self, device):
        loss_fn = CurriculumLoss().to(device)

        embeddings = torch.randn(2, 32, 256, device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)

        outputs = {"embeddings": embeddings, "mode": "continuous"}

        loss = loss_fn(outputs, targets, GenerationMode.CONTINUOUS, alpha=1.0)

        assert loss.item() == 0.0

    def test_hybrid_loss(self, device):
        loss_fn = CurriculumLoss().to(device)

        logits = torch.randn(2, 32, 1000, device=device)
        embeddings = torch.randn(2, 32, 256, device=device)
        targets = torch.randint(0, 1000, (2, 32), device=device)

        outputs = {
            "logits": logits,
            "embeddings": embeddings,
            "mode": "hybrid",
            "alpha": 0.5,
        }

        loss = loss_fn(outputs, targets, GenerationMode.HYBRID, alpha=0.5)

        assert loss.item() >= 0


class TestCurriculumIntegration:
    def test_full_curriculum_cycle(self, device):
        config = CurriculumConfig(
            discrete_steps=10, hybrid_steps=10, continuous_steps=10
        )
        scheduler = CurriculumScheduler(config)
        output_layer = HybridOutput(embed_dim=64, vocab_size=100).to(device)
        loss_fn = CurriculumLoss().to(device)

        for step in range(30):
            mode = scheduler.get_stage()
            alpha = scheduler.get_alpha()

            hidden_states = torch.randn(2, 16, 64, device=device)
            targets = torch.randint(0, 100, (2, 16), device=device)

            outputs = output_layer(hidden_states, mode, alpha, temperature=1.0)
            loss = loss_fn(outputs, targets, mode, alpha)

            assert loss.item() >= 0

            scheduler.advance()

        assert scheduler.get_stage() == GenerationMode.CONTINUOUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
