"""
Tests for Continuous Generation Module (Rectified Flow + KAN).
"""

import torch
import pytest
from neuromanifold_gpt.model.continuous.flow_scheduler import RectifiedFlowScheduler
from neuromanifold_gpt.model.continuous.velocity_field import KANVelocityField
from neuromanifold_gpt.model.continuous.output_head import ContinuousOutputHead, ContinuousOutputConfig

def test_flow_scheduler_shapes():
    scheduler = RectifiedFlowScheduler()
    B, D = 4, 16
    x1 = torch.randn(B, D)
    x0 = torch.randn(B, D)
    t = torch.rand(B)
    
    xt, v_target = scheduler.add_noise(x1, x0, t)
    
    assert xt.shape == (B, D)
    assert v_target.shape == (B, D)
    
    # Check interpolation at t=0 (should be x0)
    t_zero = torch.zeros(B)
    xt_zero, _ = scheduler.add_noise(x1, x0, t_zero)
    assert torch.allclose(xt_zero, x0, atol=1e-5)
    
    # Check interpolation at t=1 (should be x1)
    t_one = torch.ones(B)
    xt_one, _ = scheduler.add_noise(x1, x0, t_one)
    assert torch.allclose(xt_one, x1, atol=1e-5)

def test_kan_velocity_field_shapes():
    B, D = 4, 32
    model = KANVelocityField(input_dim=D, hidden_dim=64, time_dim=16)
    
    x = torch.randn(B, D)
    t = torch.rand(B)
    c = torch.randn(B, D)
    
    v = model(x, t, c)
    assert v.shape == (B, D)

def test_continuous_output_head_end_to_end():
    config = ContinuousOutputConfig(
        embed_dim=32,
        hidden_dim=64,
        num_inference_steps=5
    )
    head = ContinuousOutputHead(config)
    
    B, D = 4, 32
    target = torch.randn(B, D)
    condition = torch.randn(B, D)
    
    # Training step
    output = head.compute_loss(target, condition)
    assert 'continuous_loss' in output
    loss = output['continuous_loss']
    assert loss.ndim == 0
    
    # Inference step
    samples = head.generate(condition)
    assert samples.shape == (B, D)
    assert not torch.isnan(samples).any()

def test_gradient_flow():
    # Ensure KAN parameters receive gradients
    config = ContinuousOutputConfig(embed_dim=16, hidden_dim=32)
    head = ContinuousOutputHead(config)
    
    target = torch.randn(2, 16)
    condition = torch.randn(2, 16)
    
    output = head.compute_loss(target, condition)
    loss = output['continuous_loss']
    loss.backward()
    
    # Check spline weights in first KAN layer
    spline_grad = head.velocity_model.net[0].spline_weight.grad
    assert spline_grad is not None
    assert spline_grad.abs().sum() > 0

def test_sac_policy_shapes():
    from neuromanifold_gpt.model.continuous.sac_policy import SACPolicy, SACConfig
    config = SACConfig(state_dim=16, action_dim=16, hidden_dim=32)
    policy = SACPolicy(config)
    
    B, D = 4, 16
    state = torch.randn(B, D)
    action = torch.randn(B, D)
    
    # Test Actor
    act, log_prob, mean = policy.actor.sample(state)
    assert act.shape == (B, D)
    assert log_prob.shape == (B, 1)
    
    # Test Critic
    q1, q2 = policy.critic(state, action)
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)

def test_ddpg_policy_shapes():
    from neuromanifold_gpt.model.continuous.ddpg_policy import DDPGPolicy, DDPGConfig
    config = DDPGConfig(state_dim=16, action_dim=16, hidden_dim=32)
    policy = DDPGPolicy(config)
    
    B, D = 4, 16
    state = torch.randn(B, D)
    action = torch.randn(B, D)
    
    # Test Actor
    act = policy.get_action(state)
    assert act.shape == (B, D)
    
    # Test Critic
    q = policy.critic(state, action)
    assert q.shape == (B, 1)

