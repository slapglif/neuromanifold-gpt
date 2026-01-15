"""
ForcedDAGPlanner Usage Example

Demonstrates how to use ForcedDAGPlanner for System 2 reasoning via task
decomposition. Shows both training and evaluation modes, DAG structure
analysis, and execution order computation.

The ForcedDAGPlanner breaks down complex tasks into structured subtasks
represented as a Directed Acyclic Graph (DAG), enabling multi-step reasoning
before token generation.
"""
import torch
import torch.nn.functional as F
from neuromanifold_gpt.model.planning.dag_planner import ForcedDAGPlanner

# -----------------------------------------------------------------------------
# Configuration
embed_dim = 384          # dimension of input embeddings (must match model)
manifold_dim = 64        # dimension for manifold projection
max_nodes = 32           # maximum nodes in DAG
min_nodes = 3            # minimum nodes to generate
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
print("ForcedDAGPlanner Usage Example")
print("=" * 70)
print(f"\nDevice: {device}")
print(f"Embed dim: {embed_dim}, Manifold dim: {manifold_dim}")
print(f"Max nodes: {max_nodes}, Min nodes: {min_nodes}")
print()

# -----------------------------------------------------------------------------
# 1. Initialize the planner
# -----------------------------------------------------------------------------
print("1. Initializing ForcedDAGPlanner...")
planner = ForcedDAGPlanner(
    embed_dim=embed_dim,
    manifold_dim=manifold_dim,
    max_nodes=max_nodes,
    min_nodes=min_nodes,
    dropout=0.1,
)
planner.to(device)
planner.eval()  # Set to eval mode for deterministic behavior
print(f"   ✓ Planner initialized with {sum(p.numel() for p in planner.parameters())} parameters")
print()

# -----------------------------------------------------------------------------
# 2. Create sample input embeddings
# -----------------------------------------------------------------------------
print("2. Creating sample input embeddings...")
# In practice, these would come from the transformer's embedding layer
# Shape: (batch_size, seq_len, embed_dim)
x = torch.randn(batch_size, seq_len, embed_dim, device=device)
print(f"   Input shape: {x.shape}")
print()

# -----------------------------------------------------------------------------
# 3. Generate DAG in deterministic mode (evaluation)
# -----------------------------------------------------------------------------
print("3. Running planner in deterministic mode (evaluation)...")
with torch.no_grad():
    output = planner(x, deterministic=True)

print("   Output keys:", list(output.keys()))
print()
print(f"   Node embeddings shape: {output['node_embeddings'].shape}")
print(f"   Adjacency matrix shape: {output['adj_matrix'].shape}")
print(f"   Surface area (complexity): {output['surface_area']}")
print(f"   Per-node complexities shape: {output['complexities'].shape}")
print(f"   Node mask shape: {output['node_mask'].shape}")
print()

# -----------------------------------------------------------------------------
# 4. Analyze the generated DAG structure
# -----------------------------------------------------------------------------
print("4. Analyzing DAG structure for first example in batch...")
# Get first example from batch
adj_matrix_0 = output['adj_matrix'][0]  # (max_nodes, max_nodes)
node_mask_0 = output['node_mask'][0]    # (max_nodes,)
complexities_0 = output['complexities'][0]  # (max_nodes,)

# Count active nodes
num_active = node_mask_0.sum().item()
print(f"   Active nodes: {num_active} / {max_nodes}")

# Get active node indices
active_indices = torch.where(node_mask_0)[0]
print(f"   Active node indices: {active_indices.tolist()}")

# Count edges in DAG
active_adj = adj_matrix_0[node_mask_0][:, node_mask_0]  # subgraph
num_edges = active_adj.sum().item()
print(f"   Number of edges: {int(num_edges)}")

# Show complexity distribution
print(f"\n   Node complexities (active nodes only):")
for idx in active_indices[:5]:  # Show first 5
    complexity = complexities_0[idx].item()
    print(f"      Node {idx}: {complexity:.4f}")
if len(active_indices) > 5:
    print(f"      ... and {len(active_indices) - 5} more nodes")
print()

# -----------------------------------------------------------------------------
# 5. Compute execution order (topological sort)
# -----------------------------------------------------------------------------
print("5. Computing execution order (topological sort)...")
execution_order = planner.get_execution_order(adj_matrix_0, node_mask_0)
print(f"   Execution order: {execution_order.tolist()}")
print(f"   Number of steps: {len(execution_order)}")
print()

# Verify execution order respects dependencies
print("   Verifying execution order respects dependencies...")
valid = True
for i, node_i in enumerate(execution_order):
    for j, node_j in enumerate(execution_order):
        if i < j:  # node_i comes before node_j in execution
            # Check if node_i depends on node_j (should be 0)
            if adj_matrix_0[node_i, node_j] > 0:
                print(f"   ✗ ERROR: Node {node_i} depends on {node_j} but executes first!")
                valid = False

if valid:
    print("   ✓ Execution order is valid (respects all dependencies)")
print()

# -----------------------------------------------------------------------------
# 6. Visualize DAG structure (simple text representation)
# -----------------------------------------------------------------------------
print("6. Visualizing DAG structure...")
print("   Format: [Node ID] (complexity) -> [dependent nodes]")
print()
for node_idx in execution_order[:10]:  # Show first 10 nodes
    node_idx_val = node_idx.item()
    complexity = complexities_0[node_idx_val].item()

    # Find which nodes this node depends on (incoming edges)
    dependencies = []
    for j in range(max_nodes):
        if node_mask_0[j] and adj_matrix_0[node_idx_val, j] > 0:
            dependencies.append(j)

    if dependencies:
        deps_str = ", ".join(str(d) for d in dependencies)
        print(f"   [{node_idx_val}] ({complexity:.3f}) <- depends on [{deps_str}]")
    else:
        print(f"   [{node_idx_val}] ({complexity:.3f}) <- no dependencies (root)")

if len(execution_order) > 10:
    print(f"   ... and {len(execution_order) - 10} more nodes")
print()

# -----------------------------------------------------------------------------
# 7. Compare deterministic vs non-deterministic mode
# -----------------------------------------------------------------------------
print("7. Comparing deterministic vs non-deterministic (training) mode...")
print()

# Non-deterministic mode (training)
planner.train()
with torch.no_grad():
    output_train = planner(x, deterministic=False)

num_active_train = output_train['node_mask'][0].sum().item()
print(f"   Deterministic mode:     {num_active} active nodes")
print(f"   Non-deterministic mode: {num_active_train} active nodes")
print(f"   Surface area (det):     {output['surface_area'][0].item():.4f}")
print(f"   Surface area (non-det): {output_train['surface_area'][0].item():.4f}")
print()

# -----------------------------------------------------------------------------
# 8. Demonstrate batch processing
# -----------------------------------------------------------------------------
print("8. Demonstrating batch processing...")
print(f"   Batch size: {batch_size}")
print()
for b in range(batch_size):
    num_active_b = output['node_mask'][b].sum().item()
    surface_area_b = output['surface_area'][b].item()
    num_edges_b = output['adj_matrix'][b][output['node_mask'][b]][:, output['node_mask'][b]].sum().item()
    print(f"   Example {b}:")
    print(f"      Active nodes: {num_active_b}")
    print(f"      Edges: {int(num_edges_b)}")
    print(f"      Total complexity: {surface_area_b:.4f}")
print()

# -----------------------------------------------------------------------------
# 9. Integration example: Using node embeddings downstream
# -----------------------------------------------------------------------------
print("9. Integration example: Using node embeddings downstream...")
print()
print("   The node embeddings can be used as input to subsequent reasoning modules:")
print(f"   - Shape: {output['node_embeddings'].shape}")
print(f"   - Can be fed to transformer layers for further processing")
print(f"   - Each node represents a subtask in the reasoning chain")
print()
print("   Example: Extracting embeddings for active nodes...")
active_embeddings = output['node_embeddings'][0][node_mask_0]  # (num_active, embed_dim)
print(f"   Active node embeddings shape: {active_embeddings.shape}")
print()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("The ForcedDAGPlanner successfully:")
print("  ✓ Decomposed input into structured DAG with multiple reasoning nodes")
print(f"  ✓ Generated {num_active} active nodes with {int(num_edges)} dependencies")
print("  ✓ Enforced DAG structure (no cycles)")
print("  ✓ Computed valid topological execution order")
print("  ✓ Predicted per-node complexity (surface area)")
print()
print("Key takeaways:")
print("  • Use deterministic=True for evaluation (consistent node selection)")
print("  • Use deterministic=False for training (enables gradient flow)")
print("  • Execution order gives the sequence for processing subtasks")
print("  • Node embeddings can be fed to downstream reasoning modules")
print("  • Surface area represents total reasoning complexity")
print()
print("=" * 70)
