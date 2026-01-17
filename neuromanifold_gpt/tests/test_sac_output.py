import pytest
import torch
from neuromanifold_gpt.model.rl.sac_output import (
    GaussianActor,
    TwinQNetwork,
    SACOutputHead,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestGaussianActor:
    def test_forward_shape(self, device):
        actor = GaussianActor(state_dim=256, action_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)

        action, log_prob, mean = actor(state)

        assert action.shape == (2, 32, 256)
        assert log_prob.shape == (2, 32, 1)
        assert mean.shape == (2, 32, 256)

    def test_deterministic_mode(self, device):
        actor = GaussianActor(state_dim=256, action_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)

        action_det, log_prob_det, mean_det = actor(state, deterministic=True)

        # Deterministic action should be tanh(mean)
        assert torch.allclose(action_det, torch.tanh(mean_det), atol=1e-5)
        assert (log_prob_det == 0).all()

    def test_action_bounds(self, device):
        actor = GaussianActor(state_dim=256, action_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)

        action, _, _ = actor(state)

        assert (action >= -1).all()
        assert (action <= 1).all()


class TestTwinQNetwork:
    def test_forward_shape(self, device):
        critic = TwinQNetwork(state_dim=256, action_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)
        action = torch.randn(2, 32, 256).to(device)

        q1, q2 = critic(state, action)

        assert q1.shape == (2, 32, 1)
        assert q2.shape == (2, 32, 1)

    def test_twin_networks_different(self, device):
        critic = TwinQNetwork(state_dim=256, action_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)
        action = torch.randn(2, 32, 256).to(device)

        q1, q2 = critic(state, action)

        assert not torch.allclose(q1, q2, atol=1e-2)


class TestSACOutputHead:
    def test_initialization(self, device):
        sac = SACOutputHead(embed_dim=256, hidden_dim=128).to(device)

        assert hasattr(sac, "actor")
        assert hasattr(sac, "critic")
        assert hasattr(sac, "critic_target")
        assert hasattr(sac, "log_alpha")

    def test_select_action_shape(self, device):
        sac = SACOutputHead(embed_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)

        action = sac.select_action(state)

        assert action.shape == (2, 32, 256)

    def test_select_action_deterministic(self, device):
        sac = SACOutputHead(embed_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)

        action_det = sac.select_action(state, deterministic=True)
        action_det2 = sac.select_action(state, deterministic=True)

        assert torch.allclose(action_det, action_det2, atol=1e-5)

    def test_compute_loss(self, device):
        sac = SACOutputHead(embed_dim=256, hidden_dim=128).to(device)

        batch_size = 4
        seq_len = 16

        state = torch.randn(batch_size, seq_len, 256).to(device)
        action = torch.randn(batch_size, seq_len, 256).to(device)
        reward = torch.randn(batch_size, seq_len, 1).to(device)
        next_state = torch.randn(batch_size, seq_len, 256).to(device)
        done = torch.zeros(batch_size, seq_len, 1).to(device)

        losses = sac.compute_loss(state, action, reward, next_state, done)

        assert "actor_loss" in losses
        assert "critic_loss" in losses
        assert "alpha_loss" in losses
        assert "q_value" in losses

    def test_backward_pass(self, device):
        sac = SACOutputHead(embed_dim=128, hidden_dim=64).to(device)

        state = torch.randn(2, 16, 128).to(device)
        action = torch.randn(2, 16, 128).to(device)
        reward = torch.randn(2, 16, 1).to(device)
        next_state = torch.randn(2, 16, 128).to(device)
        done = torch.zeros(2, 16, 1).to(device)

        losses = sac.compute_loss(state, action, reward, next_state, done)

        total_loss = losses["actor_loss"] + losses["critic_loss"] + losses["alpha_loss"]
        total_loss.backward()

        actor_params_with_grad = [
            p for p in sac.actor.parameters() if p.grad is not None
        ]
        critic_params_with_grad = [
            p for p in sac.critic.parameters() if p.grad is not None
        ]

        assert len(actor_params_with_grad) > 0
        assert len(critic_params_with_grad) > 0

    def test_update_targets(self, device):
        sac = SACOutputHead(embed_dim=256, hidden_dim=128).to(device)

        target_params_before = [p.clone() for p in sac.critic_target.parameters()]

        state = torch.randn(2, 16, 256).to(device)
        action = torch.randn(2, 16, 256).to(device)
        q1, q2 = sac.critic(state, action)
        loss = q1.mean() + q2.mean()
        loss.backward()

        with torch.no_grad():
            for p in sac.critic.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        sac.update_targets()

        target_params_after = list(sac.critic_target.parameters())

        for p_before, p_after in zip(target_params_before, target_params_after):
            assert not torch.allclose(p_before, p_after, atol=1e-6)

    def test_forward_inference(self, device):
        sac = SACOutputHead(embed_dim=256, hidden_dim=128).to(device)
        state = torch.randn(2, 32, 256).to(device)

        action, info = sac(state, deterministic=True)

        assert action.shape == (2, 32, 256)
        assert "action_mean" in info
        assert "action_std" in info
        assert "log_prob" in info


class TestSACIntegration:
    def test_training_loop(self, device):
        sac = SACOutputHead(embed_dim=128, hidden_dim=64).to(device)

        actor_optimizer = torch.optim.Adam(sac.actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(sac.critic.parameters(), lr=3e-4)
        alpha_optimizer = torch.optim.Adam([sac.log_alpha], lr=3e-4)

        for _ in range(3):
            state = torch.randn(4, 16, 128).to(device)
            action = torch.randn(4, 16, 128).to(device)
            reward = torch.randn(4, 16, 1).to(device)
            next_state = torch.randn(4, 16, 128).to(device)
            done = torch.zeros(4, 16, 1).to(device)

            critic_optimizer.zero_grad()
            critic_loss = sac.compute_loss(state, action, reward, next_state, done)[
                "critic_loss"
            ]
            critic_loss.backward()
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss = sac.compute_loss(state, action, reward, next_state, done)[
                "actor_loss"
            ]
            actor_loss.backward()
            actor_optimizer.step()

            alpha_optimizer.zero_grad()
            alpha_loss = sac.compute_loss(state, action, reward, next_state, done)[
                "alpha_loss"
            ]
            alpha_loss.backward()
            alpha_optimizer.step()

            sac.update_targets()

        assert actor_loss.item() is not None
        assert critic_loss.item() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
