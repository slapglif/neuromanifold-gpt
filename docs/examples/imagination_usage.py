"""
ConsistencyImaginationModule Usage Example

Demonstrates how to use ConsistencyImaginationModule for System 2 reasoning via
counterfactual exploration. Shows both goal-free and goal-directed imagination,
scoring of alternatives, and consistency training.

The ConsistencyImaginationModule implements a "mental whiteboard" where the model
explores alternative reasoning paths in manifold space using fast consistency-based
diffusion, enabling counterfactual reasoning before token generation.
"""
import torch
import torch.nn.functional as F
from neuromanifold_gpt.model.imagination import ConsistencyImaginationModule

# -----------------------------------------------------------------------------
# Configuration
embed_dim = 384          # dimension of input embeddings (must match model)
manifold_dim = 64        # dimension for manifold exploration
n_imagination_steps = 4  # number of denoising steps
noise_scale = 1.0        # initial noise magnitude for exploration
n_alternatives = 6       # number of alternative paths to generate
batch_size = 2           # number of examples to process
seq_len = 10             # sequence length of input
seed = 1337              # random seed for reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------

# Set random seed for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print("=" * 70)
print("ConsistencyImaginationModule Usage Example")
print("=" * 70)
print(f"\nDevice: {device}")
print(f"Embed dim: {embed_dim}, Manifold dim: {manifold_dim}")
print(f"Imagination steps: {n_imagination_steps}, Noise scale: {noise_scale}")
print(f"Alternatives to generate: {n_alternatives}")
print()

# -----------------------------------------------------------------------------
# 1. Initialize the imagination module
# -----------------------------------------------------------------------------
print("1. Initializing ConsistencyImaginationModule...")
imagination = ConsistencyImaginationModule(
    embed_dim=embed_dim,
    manifold_dim=manifold_dim,
    n_imagination_steps=n_imagination_steps,
    noise_scale=noise_scale,
)
imagination.to(device)
imagination.eval()  # Set to eval mode for deterministic behavior
print(f"   ✓ Module initialized with {sum(p.numel() for p in imagination.parameters())} parameters")
print()

# -----------------------------------------------------------------------------
# 2. Create sample input embeddings
# -----------------------------------------------------------------------------
print("2. Creating sample input embeddings...")
# In practice, these would come from transformer hidden states
# Shape: (batch_size, seq_len, embed_dim)
x = torch.randn(batch_size, seq_len, embed_dim, device=device)
print(f"   Input shape: {x.shape}")
print(f"   Represents hidden states from a sequence of {seq_len} tokens")
print()

# -----------------------------------------------------------------------------
# 3. Generate alternatives without goal (free exploration)
# -----------------------------------------------------------------------------
print("3. Generating alternatives without goal (free exploration)...")
with torch.no_grad():
    output_no_goal = imagination(x, goal=None, n_alternatives=n_alternatives)

print("   Output keys:", list(output_no_goal.keys()))
print()
print(f"   Alternatives shape: {output_no_goal['alternatives'].shape}")
print(f"   Scores shape: {output_no_goal['scores'].shape}")
print(f"   Best indices: {output_no_goal['best_idx'].tolist()}")
print(f"   Best alternative shape: {output_no_goal['best_alternative'].shape}")
print()

# -----------------------------------------------------------------------------
# 4. Analyze the generated alternatives
# -----------------------------------------------------------------------------
print("4. Analyzing generated alternatives (first batch example)...")
scores_0 = output_no_goal['scores'][0]  # (n_alternatives,)
best_idx_0 = output_no_goal['best_idx'][0].item()

print(f"   Alternative scores:")
for i, score in enumerate(scores_0):
    marker = " ← BEST" if i == best_idx_0 else ""
    print(f"      Alternative {i}: {score.item():.4f}{marker}")
print()

# Compute diversity between alternatives
print("   Measuring diversity between alternatives...")
alternatives_0 = output_no_goal['alternatives'][0]  # (n_alternatives, seq_len, embed_dim)
# Pool across time dimension for comparison
alternatives_pooled = alternatives_0.mean(dim=1)  # (n_alternatives, embed_dim)

# Compute pairwise cosine similarities
similarities = []
for i in range(n_alternatives):
    for j in range(i + 1, n_alternatives):
        sim = F.cosine_similarity(
            alternatives_pooled[i].unsqueeze(0),
            alternatives_pooled[j].unsqueeze(0)
        ).item()
        similarities.append(sim)

avg_similarity = sum(similarities) / len(similarities)
print(f"   Average pairwise similarity: {avg_similarity:.4f}")
print(f"   (Lower similarity indicates more diverse exploration)")
print()

# -----------------------------------------------------------------------------
# 5. Generate alternatives with goal (goal-directed exploration)
# -----------------------------------------------------------------------------
print("5. Generating alternatives with goal (goal-directed exploration)...")
# Create a goal embedding (in practice, this could be a desired outcome representation)
goal = torch.randn(batch_size, embed_dim, device=device)
print(f"   Goal shape: {goal.shape}")
print()

with torch.no_grad():
    output_with_goal = imagination(x, goal=goal, n_alternatives=n_alternatives)

print("   Alternatives generated with goal guidance")
print(f"   Best indices: {output_with_goal['best_idx'].tolist()}")
print()

# -----------------------------------------------------------------------------
# 6. Compare goal-free vs goal-directed exploration
# -----------------------------------------------------------------------------
print("6. Comparing goal-free vs goal-directed exploration...")
print()

scores_no_goal_0 = output_no_goal['scores'][0]
scores_with_goal_0 = output_with_goal['scores'][0]

print("   Score distribution comparison (first batch):")
print(f"      No goal  - Mean: {scores_no_goal_0.mean().item():.4f}, "
      f"Std: {scores_no_goal_0.std().item():.4f}")
print(f"      With goal - Mean: {scores_with_goal_0.mean().item():.4f}, "
      f"Std: {scores_with_goal_0.std().item():.4f}")
print()

# Measure alignment with goal
print("   Measuring alignment with goal...")
goal_0 = goal[0]  # (embed_dim,)
alternatives_with_goal = output_with_goal['alternatives'][0].mean(dim=1)  # (n_alternatives, embed_dim)

goal_alignments = []
for i in range(n_alternatives):
    alignment = F.cosine_similarity(
        alternatives_with_goal[i].unsqueeze(0),
        goal_0.unsqueeze(0)
    ).item()
    goal_alignments.append(alignment)

avg_alignment = sum(goal_alignments) / len(goal_alignments)
print(f"   Average goal alignment: {avg_alignment:.4f}")
print(f"   Best alternative alignment: {goal_alignments[output_with_goal['best_idx'][0].item()]:.4f}")
print()

# -----------------------------------------------------------------------------
# 7. Demonstrate consistency loss for training
# -----------------------------------------------------------------------------
print("7. Demonstrating consistency loss for training...")
imagination.train()  # Switch to training mode

# Compute consistency loss (self-consistency)
loss_self = imagination.consistency_loss(x, target=None)
print(f"   Self-consistency loss: {loss_self.item():.4f}")
print()

# Compute consistency loss with target
target = torch.randn(batch_size, seq_len, embed_dim, device=device)
loss_target = imagination.consistency_loss(x, target=target)
print(f"   Target-based consistency loss: {loss_target.item():.4f}")
print()

print("   The consistency loss trains the network to map noisy samples")
print("   at any noise level to the same clean sample, enabling fast")
print("   one-step sampling at inference time.")
print()

# -----------------------------------------------------------------------------
# 8. Visualize the exploration process
# -----------------------------------------------------------------------------
print("8. Visualizing the counterfactual exploration process...")
print()
print("   Process flow:")
print()
print("   Input Sequence (x)")
print("       ↓ (pool over time)")
print("   Pooled Representation")
print("       ↓ (project to manifold)")
print("   Manifold Point (semantic space)")
print("       ↓ (add noise)")
print("   Noisy Samples × N (explore alternatives)")
print("       ↓ (iterative denoising)")
print("   Refined Alternatives")
print("       ↓ (score by coherence)")
print("   Best Alternative ← selected")
print()
print("   This implements a 'mental whiteboard' for exploring")
print("   what-if scenarios before committing to a token.")
print()

# -----------------------------------------------------------------------------
# 9. Demonstrate batch processing
# -----------------------------------------------------------------------------
print("9. Demonstrating batch processing...")
print(f"   Batch size: {batch_size}")
print()

imagination.eval()
with torch.no_grad():
    output_batch = imagination(x, goal=None, n_alternatives=n_alternatives)

for b in range(batch_size):
    scores_b = output_batch['scores'][b]
    best_idx_b = output_batch['best_idx'][b].item()
    best_score_b = scores_b[best_idx_b].item()
    avg_score_b = scores_b.mean().item()

    print(f"   Batch {b}:")
    print(f"      Best alternative: {best_idx_b} (score: {best_score_b:.4f})")
    print(f"      Average score: {avg_score_b:.4f}")
    print(f"      Score range: [{scores_b.min().item():.4f}, {scores_b.max().item():.4f}]")
print()

# -----------------------------------------------------------------------------
# 10. Integration example: Using imagination in reasoning
# -----------------------------------------------------------------------------
print("10. Integration example: Using imagination in reasoning pipeline...")
print()
print("   Typical usage in NeuroManifoldGPT:")
print()
print("   1. Extract hidden states from transformer layer")
print("   2. Generate N alternative continuations via imagination")
print("   3. Score alternatives by coherence and goal alignment")
print("   4. Select best alternative (or ensemble top-k)")
print("   5. Use as input to next reasoning step or token generation")
print()
print("   Example: Counterfactual reasoning")
print()

# Simulate a reasoning scenario
print("   Scenario: The model is considering how to continue a thought")
print("   Current thought: [hidden state x]")
print()

with torch.no_grad():
    # Generate alternatives
    reasoning_output = imagination(x[:1], goal=None, n_alternatives=4)

    print(f"   Generated {4} alternative continuations:")
    for i, score in enumerate(reasoning_output['scores'][0]):
        print(f"      Path {i}: coherence score = {score.item():.4f}")

    best_path_idx = reasoning_output['best_idx'][0].item()
    print()
    print(f"   ✓ Selected path {best_path_idx} as most promising continuation")
    print()

# -----------------------------------------------------------------------------
# 11. Compare different numbers of alternatives
# -----------------------------------------------------------------------------
print("11. Comparing different numbers of alternatives...")
print()

imagination.eval()
for n_alt in [2, 4, 8]:
    with torch.no_grad():
        output = imagination(x[:1], goal=None, n_alternatives=n_alt)

    scores = output['scores'][0]
    best_score = scores[output['best_idx'][0]].item()
    avg_score = scores.mean().item()

    print(f"   N={n_alt} alternatives:")
    print(f"      Best score: {best_score:.4f}")
    print(f"      Average score: {avg_score:.4f}")

print()
print("   More alternatives → better chance of finding high-quality path")
print("   Trade-off: computation time vs. exploration breadth")
print()

# -----------------------------------------------------------------------------
# 12. Demonstrate manifold exploration parameters
# -----------------------------------------------------------------------------
print("12. Demonstrating impact of exploration parameters...")
print()

# Test different noise scales
print("   a) Impact of noise scale on exploration diversity...")
for scale in [0.5, 1.0, 2.0]:
    imagination_test = ConsistencyImaginationModule(
        embed_dim=embed_dim,
        manifold_dim=manifold_dim,
        n_imagination_steps=n_imagination_steps,
        noise_scale=scale,
    )
    imagination_test.to(device)
    imagination_test.eval()

    with torch.no_grad():
        output = imagination_test(x[:1], goal=None, n_alternatives=4)

    # Measure diversity
    alts = output['alternatives'][0].mean(dim=1)
    similarities = []
    for i in range(4):
        for j in range(i + 1, 4):
            sim = F.cosine_similarity(alts[i].unsqueeze(0), alts[j].unsqueeze(0)).item()
            similarities.append(sim)
    avg_sim = sum(similarities) / len(similarities)

    print(f"      Noise scale {scale}: avg similarity = {avg_sim:.4f}")

print()
print("   Lower noise → more similar alternatives (exploitation)")
print("   Higher noise → more diverse alternatives (exploration)")
print()

# Test different numbers of denoising steps
print("   b) Impact of denoising steps on quality...")
for steps in [2, 4, 8]:
    imagination_test = ConsistencyImaginationModule(
        embed_dim=embed_dim,
        manifold_dim=manifold_dim,
        n_imagination_steps=steps,
        noise_scale=1.0,
    )
    imagination_test.to(device)
    imagination_test.eval()

    with torch.no_grad():
        output = imagination_test(x[:1], goal=None, n_alternatives=4)

    best_score = output['scores'][0][output['best_idx'][0]].item()
    print(f"      {steps} steps: best score = {best_score:.4f}")

print()
print("   More steps → better refinement but slower inference")
print("   Consistency models enable fast sampling (4-8 steps typically)")
print()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("The ConsistencyImaginationModule successfully demonstrated:")
print("  ✓ Counterfactual exploration via consistency-based diffusion")
print(f"  ✓ Generated {n_alternatives} diverse alternative reasoning paths")
print("  ✓ Goal-free exploration (free-form counterfactuals)")
print("  ✓ Goal-directed exploration (optimizing toward target)")
print("  ✓ Automatic scoring and selection of best alternative")
print("  ✓ Consistency loss for efficient training")
print("  ✓ Fast inference with few denoising steps (4-8 steps)")
print()
print("Key takeaways:")
print("  • Implements a 'mental whiteboard' for exploring what-if scenarios")
print("  • Uses manifold space for semantic exploration")
print("  • Consistency models enable fast sampling vs. traditional diffusion")
print("  • Goal conditioning guides exploration toward desired outcomes")
print("  • Trade-offs: noise scale (diversity), steps (quality), alternatives (breadth)")
print()
print("Integration tips:")
print("  • Use after planning to explore alternative execution paths")
print("  • Combine with memory to explore past experiences")
print("  • Goal can be set from user intent or learned objective")
print("  • Best alternative can be used directly or ensemble top-k")
print("  • Train with consistency loss on reasoning trajectories")
print()
print("Biological inspiration:")
print("  • Mental simulation: humans imagine alternatives before acting")
print("  • Counterfactual reasoning: 'what if I had done X instead?'")
print("  • Parallel hypothesis generation in prefrontal cortex")
print("  • Fast System 2 reasoning via efficient mental models")
print()
print("=" * 70)
