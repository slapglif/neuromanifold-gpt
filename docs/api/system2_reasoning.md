# System 2 Reasoning API

**System 2 Reasoning Components for NeuroManifoldGPT**

This document provides comprehensive API documentation for the three System 2 reasoning components that enable deliberate, structured reasoning in NeuroManifoldGPT:

1. **ForcedDAGPlanner** - Task decomposition via Directed Acyclic Graphs
2. **HierarchicalEngramMemory** - Three-tiered memory consolidation
3. **ConsistencyImaginationModule** - Counterfactual exploration

These components complement the fast, associative "System 1" reasoning of the base transformer architecture, enabling the model to engage in multi-step planning, memory consolidation, and alternative reasoning path exploration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [ForcedDAGPlanner](#forceddagplanner)
- [HierarchicalEngramMemory](#hierarchicalengrammemory)
- [ConsistencyImaginationModule](#consistencyimaginationmodule)
- [Integration Guide](#integration-guide)
- [Configuration Examples](#configuration-examples)
- [Best Practices](#best-practices)

---

## Overview

### What is System 2 Reasoning?

System 2 reasoning refers to deliberate, analytical thinking as opposed to fast, automatic System 1 processing. In the context of NeuroManifoldGPT, System 2 reasoning is implemented through three specialized neural modules:

- **Planning**: Breaking complex tasks into structured subtasks (DAG decomposition)
- **Memory**: Consolidating knowledge across temporal scales (working → short-term → long-term)
- **Imagination**: Exploring alternative reasoning paths (counterfactual simulation)

### Key Benefits

- **Structured Reasoning**: Explicit task decomposition enables multi-step problem solving
- **Long-term Memory**: Hierarchical consolidation prevents catastrophic forgetting
- **Exploration**: Counterfactual simulation discovers alternative solution paths
- **Biological Plausibility**: Inspired by hippocampus-cortex consolidation and prefrontal planning

### When to Use

System 2 reasoning components are optional and should be enabled when:

- Tasks require multi-step reasoning or planning
- Long-context scenarios benefit from hierarchical memory
- Exploring multiple solution paths improves robustness
- Training data includes complex, compositional problems

For simple, single-step tasks, the base transformer architecture may be more efficient.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     NeuroManifoldGPT Model                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐  │
│  │ Input Tokens    │───▶│ Semantic Folding Encoder (SDR)   │  │
│  └─────────────────┘    └──────────────────────────────────┘  │
│                                        │                        │
│                                        ▼                        │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │            Transformer Blocks (System 1)                │  │
│  │  - Manifold-Spectral Attention                          │  │
│  │  - FHN Soliton Dynamics                                 │  │
│  │  - KAN-based MLPs                                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           │                                    │
│         ┌─────────────────┼─────────────────┐                 │
│         │                 │                 │                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   DAG       │  │Hierarchical │  │Imagination  │           │
│  │  Planner    │  │   Memory    │  │   Module    │           │
│  │ (Planning)  │  │  (Memory)   │  │(Exploration)│           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                 │
│                           ▼                                    │
│                  ┌─────────────────┐                          │
│                  │   LM Head       │                          │
│                  │ (Token Logits)  │                          │
│                  └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input** → SDR encoding → Transformer blocks
2. **System 1** (Fast): Manifold-spectral attention processes tokens
3. **System 2** (Slow): Optional routing to reasoning modules
   - DAG Planner: Decomposes task into subtask graph
   - Memory: Stores/retrieves from hierarchical tiers
   - Imagination: Generates alternative reasoning paths
4. **Output** → Language model head → Token predictions

---

## ForcedDAGPlanner

### Overview

The **ForcedDAGPlanner** decomposes complex tasks into Directed Acyclic Graphs (DAGs) for systematic multi-step reasoning. It transforms an input embedding sequence into a structured graph of subtasks with explicit dependencies, enabling the model to "think before answering."

**Key Insight**: Complex reasoning requires breaking problems into subtasks. The DAG structure enforces logical dependencies while preventing circular reasoning.

### Class Definition

```python
class ForcedDAGPlanner(nn.Module):
    """
    Decomposes tasks into DAGs for structured reasoning.

    Args:
        embed_dim: Dimension of input embeddings (typically 384)
        manifold_dim: Dimension for manifold projection (default 64)
        max_nodes: Maximum number of nodes in DAG (default 32)
        min_nodes: Minimum number of nodes to generate (default 3)
        dropout: Dropout probability (default 0.0)
    """
```

### Architecture Components

1. **Task Encoder**: Aggregates sequence into single task representation via attention pooling
2. **Node Generator**: Predicts N subtask nodes via learned manifold projection
3. **Edge Predictor**: Computes pairwise dependencies (adjacency matrix)
4. **DAG Enforcer**: Masks edges to ensure acyclicity via triangular constraint
5. **Complexity Estimator**: Predicts surface area (information content) of each node

### API Reference

#### Constructor

```python
planner = ForcedDAGPlanner(
    embed_dim=384,           # Must match model embedding dimension
    manifold_dim=64,         # Manifold space for geometric reasoning
    max_nodes=32,            # Maximum subtasks in DAG
    min_nodes=3,             # Minimum subtasks to enforce
    dropout=0.0,             # Dropout for regularization
)
```

#### Forward Method

```python
def forward(
    self,
    x: torch.Tensor,           # (B, T, embed_dim) input embeddings
    deterministic: bool = False # Use argmax for node selection (eval)
) -> dict[str, torch.Tensor]:
    """
    Decompose input into task DAG.

    Args:
        x: Input embeddings (batch_size, seq_len, embed_dim)
        deterministic: If True, use argmax for node selection (eval mode)

    Returns:
        Dictionary containing:
            - node_embeddings: (B, max_nodes, embed_dim) node representations
            - adj_matrix: (B, max_nodes, max_nodes) adjacency matrix (DAG)
            - surface_area: (B,) total surface area (sum of complexities)
            - complexities: (B, max_nodes) per-node complexity scores
            - node_mask: (B, max_nodes) boolean mask of active nodes
    """
```

**Example:**

```python
import torch
from neuromanifold_gpt.model.planning.dag_planner import ForcedDAGPlanner

# Initialize planner
planner = ForcedDAGPlanner(embed_dim=384, manifold_dim=64)
planner.eval()

# Create input embeddings (batch=2, seq_len=10, embed_dim=384)
x = torch.randn(2, 10, 384)

# Generate DAG
with torch.no_grad():
    output = planner(x, deterministic=True)

# Access outputs
node_embeddings = output['node_embeddings']  # (2, 32, 384)
adj_matrix = output['adj_matrix']            # (2, 32, 32)
surface_area = output['surface_area']        # (2,)
node_mask = output['node_mask']              # (2, 32)

# Count active nodes
num_active = node_mask[0].sum().item()
print(f"Active nodes: {num_active}")
```

#### Execution Order Method

```python
def get_execution_order(
    self,
    adj_matrix: torch.Tensor,  # (max_nodes, max_nodes) adjacency
    node_mask: torch.Tensor    # (max_nodes,) active nodes mask
) -> torch.Tensor:
    """
    Compute topological ordering of DAG (Kahn's algorithm).

    Returns the order in which nodes should be executed, respecting
    dependency constraints. Nodes with no dependencies come first.

    Args:
        adj_matrix: (max_nodes, max_nodes) adjacency matrix
                   adj[i,j]=1 means i depends on j (j before i)
        node_mask: (max_nodes,) boolean mask of active nodes

    Returns:
        execution_order: (num_active_nodes,) indices in execution order
    """
```

**Example:**

```python
# Compute execution order for first batch element
adj = output['adj_matrix'][0]
mask = output['node_mask'][0]
order = planner.get_execution_order(adj, mask)

print(f"Execution order: {order.tolist()}")
# Example output: [5, 12, 8, 3, 17, 9]
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | int | - | Input embedding dimension (must match model) |
| `manifold_dim` | int | 64 | Manifold space dimension for geometric reasoning |
| `max_nodes` | int | 32 | Maximum subtasks in DAG |
| `min_nodes` | int | 3 | Minimum subtasks to enforce |
| `dropout` | float | 0.0 | Dropout probability for regularization |

### Model Configuration

Enable in NeuroManifoldConfig:

```python
config = NeuroManifoldConfig(
    use_dag_planner=True,
    dag_max_nodes=32,
    dag_min_nodes=3,
)
```

### DAG Structure

The adjacency matrix represents dependencies:

```
adj[i, j] = 1  =>  Node i depends on Node j
                   (j must execute before i)
```

**DAG Enforcement**: The upper triangular mask ensures acyclicity:
- Only connections where `i < j` are allowed
- Execution flows from high indices to low indices
- No cycles are possible by construction

### Output Specification

| Output Key | Shape | Description |
|------------|-------|-------------|
| `node_embeddings` | `(B, max_nodes, embed_dim)` | Embedding for each subtask node |
| `adj_matrix` | `(B, max_nodes, max_nodes)` | Binary adjacency matrix (DAG) |
| `surface_area` | `(B,)` | Total complexity (sum of node complexities) |
| `complexities` | `(B, max_nodes)` | Per-node complexity (information content) |
| `node_mask` | `(B, max_nodes)` | Boolean mask indicating active nodes |

### Use Cases

1. **Multi-step Math Problems**: Decompose "solve for x" into subtasks
2. **Code Generation**: Break down "implement feature X" into functions
3. **Reasoning Chains**: Structure "why did Y happen?" into causal steps
4. **Planning**: Transform "achieve goal G" into action sequence

### Example: Complete Workflow

```python
"""Complete DAG Planner workflow example."""
import torch
from neuromanifold_gpt.model.planning.dag_planner import ForcedDAGPlanner

# Initialize
planner = ForcedDAGPlanner(embed_dim=384, manifold_dim=64, max_nodes=16)
planner.eval()

# Input: embeddings from transformer
x = torch.randn(1, 20, 384)  # (1 example, 20 tokens, 384 dims)

# Generate DAG
with torch.no_grad():
    dag = planner(x, deterministic=True)

# Analyze structure
adj = dag['adj_matrix'][0]
mask = dag['node_mask'][0]
active_nodes = torch.where(mask)[0]

print(f"Generated {len(active_nodes)} subtasks")

# Get execution order
order = planner.get_execution_order(adj, mask)
print(f"Execution sequence: {order.tolist()}")

# Show dependencies for each node
for node_id in order[:5]:  # First 5 nodes
    deps = torch.where(adj[node_id] > 0)[0]
    complexity = dag['complexities'][0][node_id].item()
    print(f"Node {node_id}: complexity={complexity:.3f}, depends_on={deps.tolist()}")

# Use node embeddings downstream
node_embeds = dag['node_embeddings'][0][mask]  # (num_active, 384)
# Feed to transformer layers or other reasoning modules
```

**See Also**: [`docs/examples/dag_planner_usage.py`](../examples/dag_planner_usage.py) for comprehensive examples.

---

## HierarchicalEngramMemory

### Overview

The **HierarchicalEngramMemory** implements a three-tiered memory hierarchy inspired by biological memory consolidation:

- **L1 (Working Memory)**: Fast access, small capacity, recent memories
- **L2 (Short-term Memory)**: Medium capacity, frequently accessed
- **L3 (Long-term Memory)**: Large capacity, consolidated knowledge

This mirrors the hippocampus → cortex consolidation pathway, where memories transition from fast-access short-term storage to stable long-term storage over time.

**Key Insight**: Biological memory systems use hierarchical consolidation to balance fast access with large capacity, preventing catastrophic forgetting.

### Class Definition

```python
class HierarchicalEngramMemory(nn.Module):
    """
    Three-tiered hierarchical memory using SDR overlap for retrieval.

    Args:
        sdr_size: Size of SDR vectors (typically 2048)
        n_active: Number of active bits in SDRs (for similarity normalization)
        content_dim: Dimension of content vectors (default 384)
        l1_capacity: Working memory capacity (default 64)
        l2_capacity: Short-term memory capacity (default 512)
        l3_capacity: Long-term memory capacity (default 4096)
        threshold: Minimum similarity for retrieval (default 0.3)
    """
```

### Architecture Components

1. **L1 (Working Memory)**: Circular buffer, most recent memories, checked first
2. **L2 (Short-term Memory)**: Receives L1 evictions, medium-term storage
3. **L3 (Long-term Memory)**: Receives L2 evictions, consolidated knowledge
4. **Retrieval**: Searches L1 → L2 → L3, combines results by similarity

### Memory Flow

```
New Memory
    ↓
┌───────┐
│  L1   │ (Working Memory - 64 slots)
│ Fast  │ - Circular buffer
└───┬───┘ - Most recent
    │ Eviction when full
    ↓
┌───────┐
│  L2   │ (Short-term Memory - 512 slots)
│Medium │ - Recent history
└───┬───┘ - Frequently accessed
    │ Eviction when full
    ↓
┌───────┐
│  L3   │ (Long-term Memory - 4096 slots)
│ Slow  │ - Consolidated knowledge
└───────┘ - Stable storage

Retrieval: L1 → L2 → L3 (combine top-k from each)
```

### API Reference

#### Constructor

```python
memory = HierarchicalEngramMemory(
    sdr_size=2048,           # SDR vector size
    n_active=40,             # Active bits per SDR
    content_dim=384,         # Content vector dimension
    l1_capacity=64,          # Working memory size
    l2_capacity=512,         # Short-term memory size
    l3_capacity=4096,        # Long-term memory size
    threshold=0.3,           # Retrieval similarity threshold
)
```

#### Store Method

```python
def store(
    self,
    sdr: torch.Tensor,      # (sdr_size,) SDR marker
    content: torch.Tensor   # (content_dim,) content vector
) -> None:
    """
    Store SDR-content pair in L1 (working memory).

    If L1 is full, the oldest entry is evicted to L2 before storing.
    This maintains the memory hierarchy: new memories always enter
    at the top (L1) and flow down through the tiers over time.
    """
```

**Example:**

```python
import torch
from neuromanifold_gpt.model.memory.hierarchical_engram import HierarchicalEngramMemory

# Initialize memory
memory = HierarchicalEngramMemory(
    sdr_size=2048,
    n_active=40,
    content_dim=384,
    l1_capacity=64,
    l2_capacity=512,
    l3_capacity=4096,
)

# Create sparse SDR (40 active bits out of 2048)
indices = torch.randperm(2048)[:40]
sdr = torch.zeros(2048)
sdr[indices] = 1.0

# Create content vector
content = torch.randn(384)

# Store in memory (enters L1)
memory.store(sdr, content)

# Check tier sizes
sizes = memory.get_size()
print(f"L1: {sizes['l1'].item()}, L2: {sizes['l2'].item()}, L3: {sizes['l3'].item()}")
```

#### Store Batch Method

```python
def store_batch(
    self,
    sdrs: torch.Tensor,      # (N, sdr_size) batch of SDRs
    contents: torch.Tensor   # (N, content_dim) batch of contents
) -> None:
    """
    Store multiple SDR-content pairs efficiently (vectorized).

    All memories are stored in L1, triggering cascading evictions
    to L2 and L3 as needed.
    """
```

**Example:**

```python
# Store multiple memories at once
batch_size = 10
sdrs = torch.zeros(batch_size, 2048)
for i in range(batch_size):
    indices = torch.randperm(2048)[:40]
    sdrs[i, indices] = 1.0

contents = torch.randn(batch_size, 384)

memory.store_batch(sdrs, contents)
```

#### Retrieve Method

```python
def retrieve(
    self,
    query_sdr: torch.Tensor,  # (sdr_size,) query SDR
    top_k: int = 5             # Maximum results to return
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """
    Retrieve by SDR similarity across all tiers.

    Searches L1 → L2 → L3 in order, combining results.
    Gives priority to recent memories (L1) while still
    accessing consolidated knowledge (L2, L3).

    Returns:
        contents: (n_results, content_dim) retrieved content vectors
        similarities: (n_results,) overlap-based similarity scores
        tier_stats: dict with number of results from each tier
    """
```

**Example:**

```python
# Create query SDR (similar to stored SDR)
query_indices = torch.randperm(2048)[:40]
query_sdr = torch.zeros(2048)
query_sdr[query_indices] = 1.0

# Retrieve similar memories
contents, similarities, tier_stats = memory.retrieve(query_sdr, top_k=5)

print(f"Retrieved {len(contents)} memories")
print(f"From L1: {tier_stats['l1']}, L2: {tier_stats['l2']}, L3: {tier_stats['l3']}")
print(f"Similarities: {similarities.tolist()}")

# Use retrieved contents
if len(contents) > 0:
    # Blend with current state
    avg_content = contents.mean(dim=0)  # (content_dim,)
```

#### Utility Methods

```python
def get_size(self) -> dict[str, torch.Tensor]:
    """Return number of memories in each tier."""

def clear(self) -> None:
    """Clear all memories from all tiers."""

def __len__(self) -> int:
    """Return total number of stored memories."""
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sdr_size` | int | 2048 | Size of SDR vectors |
| `n_active` | int | 40 | Number of active bits in SDRs |
| `content_dim` | int | 384 | Dimension of content vectors |
| `l1_capacity` | int | 64 | Working memory capacity (fast) |
| `l2_capacity` | int | 512 | Short-term memory capacity (medium) |
| `l3_capacity` | int | 4096 | Long-term memory capacity (large) |
| `threshold` | float | 0.3 | Minimum similarity for retrieval |

### Model Configuration

Enable in NeuroManifoldConfig:

```python
config = NeuroManifoldConfig(
    use_hierarchical_memory=True,
    hierarchical_l1_capacity=64,
    hierarchical_l2_capacity=512,
    hierarchical_l3_capacity=4096,
)
```

### Retrieval Mechanism

**SDR Overlap Similarity**:

```
similarity = (query_sdr · stored_sdr) / n_active
```

- Measures bit overlap between query and stored SDRs
- Normalized by `n_active` to give similarity ∈ [0, 1]
- Threshold filters out low-similarity matches

**Hierarchical Search**:

1. Search L1 (working memory) first → fast path for recent memories
2. Search L2 (short-term) → medium speed for recent history
3. Search L3 (long-term) → comprehensive for consolidated knowledge
4. Combine results, sort by similarity, return top-k

### Use Cases

1. **Long Context**: Store distant context for retrieval beyond attention window
2. **Episodic Memory**: Remember specific examples from training/inference
3. **Knowledge Base**: Build persistent knowledge across conversations
4. **Continual Learning**: Prevent catastrophic forgetting via consolidation

### Example: Complete Workflow

```python
"""Complete Hierarchical Memory workflow example."""
import torch
from neuromanifold_gpt.model.memory.hierarchical_engram import HierarchicalEngramMemory

# Initialize with realistic capacities
memory = HierarchicalEngramMemory(
    sdr_size=2048,
    n_active=40,
    content_dim=384,
    l1_capacity=64,
    l2_capacity=512,
    l3_capacity=4096,
    threshold=0.3,
)

# Helper: Create random SDR
def make_sdr(seed):
    torch.manual_seed(seed)
    indices = torch.randperm(2048)[:40]
    sdr = torch.zeros(2048)
    sdr[indices] = 1.0
    return sdr

# Store 100 memories (will fill L1, cascade to L2, L3)
print("Storing 100 memories...")
for i in range(100):
    sdr = make_sdr(i)
    content = torch.randn(384)
    memory.store(sdr, content)

# Check distribution across tiers
sizes = memory.get_size()
print(f"L1: {sizes['l1'].item()}, L2: {sizes['l2'].item()}, L3: {sizes['l3'].item()}")
# Expected: L1=64 (full), L2=36 (overflow from L1), L3=0 (not filled yet)

# Query with a similar SDR (modify first stored SDR slightly)
query_sdr = make_sdr(0)
# Flip 5 bits to simulate partial match
flip_indices = torch.randperm(40)[:5]
active_indices = torch.where(query_sdr > 0)[0]
query_sdr[active_indices[flip_indices]] = 0.0
new_indices = torch.randperm(2048)[40:45]
query_sdr[new_indices] = 1.0

# Retrieve
contents, sims, tier_stats = memory.retrieve(query_sdr, top_k=5)

print(f"\nRetrieved {len(contents)} memories:")
print(f"  From L1: {tier_stats['l1']}")
print(f"  From L2: {tier_stats['l2']}")
print(f"  From L3: {tier_stats['l3']}")
print(f"  Similarities: {[f'{s:.3f}' for s in sims.tolist()]}")

# Use retrieved memory
if len(contents) > 0:
    best_match = contents[0]  # Highest similarity
    print(f"\nBest match shape: {best_match.shape}")
```

**See Also**: [`docs/examples/hierarchical_memory_usage.py`](../examples/hierarchical_memory_usage.py) for comprehensive examples.

---

## ConsistencyImaginationModule

### Overview

The **ConsistencyImaginationModule** uses consistency models (fast diffusion) to explore alternative reasoning paths in manifold space. It implements a "mental whiteboard" where the model can imagine different ways a thought sequence could unfold.

**Key Insight**: Human reasoning involves counterfactual simulation - imagining "what if?" scenarios. Consistency models enable fast generation of alternatives via learned consistency functions.

### Class Definition

```python
class ConsistencyImaginationModule(nn.Module):
    """
    Consistency model for exploring alternative reasoning paths.

    Args:
        embed_dim: Dimension of input embeddings (typically 384)
        manifold_dim: Dimension of manifold space (typically 64)
        n_imagination_steps: Number of denoising steps (default 4)
        noise_scale: Initial noise magnitude for exploration (default 1.0)
    """
```

### Architecture Components

1. **Manifold Projection**: Projects embeddings to manifold space for exploration
2. **Consistency Network**: Predicts clean samples from noisy inputs (denoising)
3. **Goal Encoder**: Optional goal conditioning for directed exploration
4. **Scorer**: Evaluates coherence of generated alternatives
5. **Back Projection**: Projects manifold samples back to embedding space

### Imagination Process

```
Input State (x)
      ↓
 Project to Manifold
      ↓
 Add Noise (explore nearby regions)
      ↓
 Iterative Denoising (n_steps)
 - Time embedding
 - Consistency function
 - Optional goal guidance
      ↓
 Multiple Alternatives Generated
      ↓
 Score by Coherence
      ↓
 Select Best Alternative
```

### API Reference

#### Constructor

```python
imagination = ConsistencyImaginationModule(
    embed_dim=384,           # Input embedding dimension
    manifold_dim=64,         # Manifold space for exploration
    n_imagination_steps=4,   # Denoising iterations
    noise_scale=1.0,         # Initial noise magnitude
)
```

#### Forward Method

```python
def forward(
    self,
    x: torch.Tensor,              # (B, T, embed_dim) input states
    goal: torch.Tensor | None = None,  # (B, embed_dim) optional goal
    n_alternatives: int = 4       # Number of paths to generate
) -> dict[str, torch.Tensor]:
    """
    Generate alternative reasoning paths via consistency-based exploration.

    Args:
        x: Input hidden states (B, T, embed_dim)
        goal: Optional goal embedding (B, embed_dim) to guide exploration
        n_alternatives: Number of alternative paths to generate

    Returns:
        dict containing:
            - alternatives: (B, n_alternatives, T, embed_dim) generated paths
            - scores: (B, n_alternatives) coherence scores for each
            - best_idx: (B,) index of highest-scoring alternative
            - best_alternative: (B, T, embed_dim) best path per batch
    """
```

**Example:**

```python
import torch
from neuromanifold_gpt.model.imagination import ConsistencyImaginationModule

# Initialize imagination module
imagination = ConsistencyImaginationModule(
    embed_dim=384,
    manifold_dim=64,
    n_imagination_steps=4,
    noise_scale=1.0,
)
imagination.eval()

# Input: current hidden state
x = torch.randn(2, 10, 384)  # (batch=2, seq_len=10, embed_dim=384)

# Optional: goal state to guide exploration
goal = torch.randn(2, 384)   # (batch=2, embed_dim=384)

# Generate alternatives
with torch.no_grad():
    output = imagination(x, goal=goal, n_alternatives=4)

# Access outputs
alternatives = output['alternatives']      # (2, 4, 10, 384)
scores = output['scores']                  # (2, 4)
best_idx = output['best_idx']              # (2,)
best_alternative = output['best_alternative']  # (2, 10, 384)

print(f"Generated {alternatives.shape[1]} alternatives")
print(f"Best alternative indices: {best_idx.tolist()}")
print(f"Scores: {scores[0].tolist()}")
```

#### Training Loss Method

```python
def consistency_loss(
    self,
    x: torch.Tensor,                  # (B, T, embed_dim) input samples
    target: torch.Tensor | None = None  # (B, T, embed_dim) target
) -> torch.Tensor:
    """
    Compute consistency training loss.

    The consistency loss trains the network to map noisy samples at any
    noise level to the same clean sample. This enables one-step sampling.

    Args:
        x: Input samples (B, T, embed_dim)
        target: Optional target clean samples (B, T, embed_dim)
               If None, uses x as target (self-consistency)

    Returns:
        loss: Scalar consistency loss
    """
```

**Example:**

```python
# Training mode: compute consistency loss
imagination.train()

# Sample batch
x = torch.randn(4, 10, 384, requires_grad=True)

# Compute consistency loss (self-consistency)
loss = imagination.consistency_loss(x, target=None)

# Or with explicit target
target = torch.randn(4, 10, 384)
loss = imagination.consistency_loss(x, target=target)

# Backprop
loss.backward()
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | int | 384 | Input embedding dimension |
| `manifold_dim` | int | 64 | Manifold space dimension |
| `n_imagination_steps` | int | 4 | Number of denoising iterations |
| `noise_scale` | float | 1.0 | Initial noise magnitude for exploration |

### Model Configuration

Enable in NeuroManifoldConfig:

```python
config = NeuroManifoldConfig(
    use_imagination=True,
    imagination_steps=4,
    imagination_n_alternatives=4,
)
```

### Consistency Model Theory

**Consistency Function**: A function that maps noisy samples at any noise level to the same clean sample:

```
f(x + t·noise, t) = x_clean  for all t ∈ [0, 1]
```

**Benefits**:
- Fast sampling: No need for many denoising steps (unlike DDPM)
- Stable training: Consistency constraint provides strong supervision
- Exploration: Noise allows discovering alternatives in manifold space

**Denoising Process**:

```python
for step in range(n_imagination_steps):
    t = 1.0 - (step / n_imagination_steps)  # Decrease from 1 to 0
    denoised = consistency_net(noisy_sample, t)
    if goal is not None:
        # Blend toward goal (stronger as t→0)
        goal_weight = (step + 1) / n_imagination_steps
        denoised = (1 - goal_weight) * denoised + goal_weight * goal
    noisy_sample = denoised
```

### Use Cases

1. **Beam Search**: Generate multiple reasoning paths, select best
2. **Counterfactual Reasoning**: "What if we tried approach B instead?"
3. **Robustness**: Explore alternatives to avoid local minima
4. **Planning**: Simulate different action sequences before committing

### Output Specification

| Output Key | Shape | Description |
|------------|-------|-------------|
| `alternatives` | `(B, n_alternatives, T, embed_dim)` | Generated alternative paths |
| `scores` | `(B, n_alternatives)` | Coherence score for each alternative |
| `best_idx` | `(B,)` | Index of highest-scoring alternative |
| `best_alternative` | `(B, T, embed_dim)` | Best alternative per batch element |

### Example: Complete Workflow

```python
"""Complete Imagination Module workflow example."""
import torch
from neuromanifold_gpt.model.imagination import ConsistencyImaginationModule

# Initialize
imagination = ConsistencyImaginationModule(
    embed_dim=384,
    manifold_dim=64,
    n_imagination_steps=4,
    noise_scale=1.0,
)

# Evaluation: Generate alternatives
imagination.eval()
x = torch.randn(1, 20, 384)  # (1 example, 20 tokens, 384 dims)

# Without goal: explore freely
with torch.no_grad():
    result = imagination(x, goal=None, n_alternatives=8)

print(f"Generated {result['alternatives'].shape[1]} alternatives")
print(f"Scores: {result['scores'][0].tolist()}")

# Select best alternative
best = result['best_alternative']  # (1, 20, 384)
print(f"Best alternative shape: {best.shape}")

# With goal: directed exploration
goal = torch.randn(1, 384)  # Target state
with torch.no_grad():
    result_goal = imagination(x, goal=goal, n_alternatives=8)

print(f"\nWith goal guidance:")
print(f"Scores: {result_goal['scores'][0].tolist()}")

# Training: Consistency loss
imagination.train()
x_train = torch.randn(4, 15, 384, requires_grad=True)
target_train = torch.randn(4, 15, 384)

loss = imagination.consistency_loss(x_train, target=target_train)
print(f"\nConsistency loss: {loss.item():.4f}")

# Integrate with transformer
# - Use imagination to generate alternatives at each layer
# - Select best alternative based on downstream task loss
# - This implements a form of "thinking before acting"
```

**See Also**: [`docs/examples/imagination_usage.py`](../examples/imagination_usage.py) for comprehensive examples.

---

## Integration Guide

### Enabling in Model Config

All three System 2 reasoning components are optional and controlled via `NeuroManifoldConfig`:

```python
from neuromanifold_gpt.config import NeuroManifoldConfig

config = NeuroManifoldConfig(
    # Base model parameters
    vocab_size=50257,
    n_layer=12,
    n_embd=384,
    n_heads=8,

    # System 2 Reasoning Components
    # 1. DAG Planner (Task Decomposition)
    use_dag_planner=True,
    dag_max_nodes=32,
    dag_min_nodes=3,

    # 2. Hierarchical Memory (L1/L2/L3 Consolidation)
    use_hierarchical_memory=True,
    hierarchical_l1_capacity=64,
    hierarchical_l2_capacity=512,
    hierarchical_l3_capacity=4096,

    # 3. Imagination (Counterfactual Exploration)
    use_imagination=True,
    imagination_steps=4,
    imagination_n_alternatives=4,
)

# Initialize model with System 2 reasoning
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
model = NeuroManifoldGPT(config)
```

### Accessing Components

System 2 components are accessible as model attributes when enabled:

```python
# Check if components are enabled
if model.use_dag_planner:
    dag_output = model.dag_planner(hidden_states, deterministic=False)

if model.use_hierarchical_memory:
    model.hierarchical_memory.store(sdr, content)
    retrieved, sims, stats = model.hierarchical_memory.retrieve(query_sdr)

if model.use_imagination:
    alternatives = model.imagination(hidden_states, goal=None, n_alternatives=4)
```

### Forward Pass Integration

The components integrate into the forward pass as follows:

```python
def forward(self, tokens, targets=None):
    # 1. Standard transformer processing (System 1)
    x = self.embed(tokens)
    for block in self.blocks:
        x = block(x)

    # 2. Optional System 2 reasoning
    if self.use_dag_planner:
        # Decompose task into DAG
        dag = self.dag_planner(x, deterministic=not self.training)
        # Use dag['node_embeddings'] for structured reasoning

    if self.use_hierarchical_memory:
        # Store current state in memory
        sdr = self.to_sdr(x)
        self.hierarchical_memory.store(sdr[-1], x[-1])
        # Retrieve relevant context
        retrieved, _, _ = self.hierarchical_memory.retrieve(sdr[0])
        if len(retrieved) > 0:
            # Blend retrieved memories with current state
            x = x + 0.1 * retrieved.mean(dim=0).unsqueeze(0)

    if self.use_imagination:
        # Explore alternative paths
        alts = self.imagination(x, n_alternatives=4)
        # Use best_alternative for prediction
        x = alts['best_alternative']

    # 3. LM head
    logits = self.lm_head(x)
    return logits
```

### Training Considerations

**Gradients**: All components support backpropagation:
- DAG Planner: Uses Gumbel-Softmax for differentiable node selection
- Memory: Retrieval is differentiable via soft attention
- Imagination: Consistency network trains with MSE loss

**Loss Weighting**:

```python
# Main language modeling loss
lm_loss = F.cross_entropy(logits, targets)

# Optional auxiliary losses
dag_loss = dag['surface_area'].mean()  # Encourage simple DAGs
imagination_loss = imagination.consistency_loss(x)  # Train consistency

# Combined loss
total_loss = lm_loss + 0.01 * dag_loss + 0.1 * imagination_loss
```

### Inference Modes

**Deterministic Evaluation**:

```python
model.eval()
with torch.no_grad():
    # DAG planner uses argmax for node selection
    dag = model.dag_planner(x, deterministic=True)
    # Imagination explores systematically
    alts = model.imagination(x, n_alternatives=8)
```

**Stochastic Training**:

```python
model.train()
# DAG planner samples nodes stochastically
dag = model.dag_planner(x, deterministic=False)
# Imagination adds noise for exploration
alts = model.imagination(x, n_alternatives=4)
```

### Memory Management

**Clearing Memory**:

```python
# Clear hierarchical memory between episodes
model.hierarchical_memory.clear()

# Or clear specific tier
model.hierarchical_memory.l1_count.zero_()
```

**Memory Statistics**:

```python
# Check memory usage
sizes = model.hierarchical_memory.get_size()
total = sum(sizes.values())
print(f"Total memories: {total}")
print(f"L1: {sizes['l1']}, L2: {sizes['l2']}, L3: {sizes['l3']}")
```

---

## Configuration Examples

### Minimal Configuration

Smallest System 2 setup for experimentation:

```python
config = NeuroManifoldConfig(
    vocab_size=50257,
    n_layer=6,
    n_embd=256,
    n_heads=4,

    # Enable only DAG planner
    use_dag_planner=True,
    dag_max_nodes=16,
    dag_min_nodes=2,

    # Disable other components
    use_hierarchical_memory=False,
    use_imagination=False,
)
```

### Balanced Configuration

Good balance of performance and efficiency:

```python
config = NeuroManifoldConfig(
    vocab_size=50257,
    n_layer=12,
    n_embd=384,
    n_heads=8,

    # All System 2 components enabled
    use_dag_planner=True,
    dag_max_nodes=24,
    dag_min_nodes=3,

    use_hierarchical_memory=True,
    hierarchical_l1_capacity=64,
    hierarchical_l2_capacity=256,
    hierarchical_l3_capacity=2048,

    use_imagination=True,
    imagination_steps=4,
    imagination_n_alternatives=4,
)
```

### Maximum Reasoning

Maximum System 2 reasoning capacity:

```python
config = NeuroManifoldConfig(
    vocab_size=50257,
    n_layer=24,
    n_embd=768,
    n_heads=12,

    # Maximum DAG capacity
    use_dag_planner=True,
    dag_max_nodes=64,
    dag_min_nodes=5,

    # Large hierarchical memory
    use_hierarchical_memory=True,
    hierarchical_l1_capacity=128,
    hierarchical_l2_capacity=1024,
    hierarchical_l3_capacity=8192,

    # Extensive imagination
    use_imagination=True,
    imagination_steps=8,
    imagination_n_alternatives=8,
)
```

### Task-Specific Configurations

**Math/Code (Planning Focus)**:

```python
config = NeuroManifoldConfig(
    # Emphasize DAG planning
    use_dag_planner=True,
    dag_max_nodes=32,  # Many subtasks for complex problems

    use_hierarchical_memory=False,  # Less important for math
    use_imagination=True,
    imagination_n_alternatives=4,  # Explore alternative solutions
)
```

**Long Context (Memory Focus)**:

```python
config = NeuroManifoldConfig(
    use_dag_planner=False,  # Less important for retrieval

    # Large hierarchical memory
    use_hierarchical_memory=True,
    hierarchical_l1_capacity=128,
    hierarchical_l2_capacity=2048,
    hierarchical_l3_capacity=16384,

    use_imagination=False,  # Less important for recall
)
```

**Creative Tasks (Imagination Focus)**:

```python
config = NeuroManifoldConfig(
    use_dag_planner=True,
    dag_max_nodes=16,  # Light planning

    use_hierarchical_memory=True,
    hierarchical_l1_capacity=64,
    hierarchical_l2_capacity=512,
    hierarchical_l3_capacity=4096,

    # Extensive exploration
    use_imagination=True,
    imagination_steps=8,
    imagination_n_alternatives=16,  # Many alternatives
)
```

---

## Best Practices

### When to Enable System 2 Reasoning

**✓ Good Use Cases**:
- Multi-step reasoning tasks (math, logic, planning)
- Long-context scenarios (>2048 tokens)
- Tasks requiring exploration (creative writing, code generation)
- Continual learning scenarios (prevent catastrophic forgetting)

**✗ Avoid For**:
- Simple classification tasks
- Short sequences (<512 tokens)
- Real-time inference with strict latency requirements
- Tasks with abundant training data (System 1 may be sufficient)

### Performance Optimization

**Memory Efficiency**:

```python
# Use smaller capacities during initial experiments
config.hierarchical_l1_capacity = 32  # Instead of 64
config.hierarchical_l2_capacity = 256  # Instead of 512
config.hierarchical_l3_capacity = 2048  # Instead of 4096

# Reduce DAG size
config.dag_max_nodes = 16  # Instead of 32
```

**Inference Speed**:

```python
# Reduce imagination alternatives
config.imagination_n_alternatives = 2  # Instead of 4
config.imagination_steps = 2  # Instead of 4

# Use deterministic mode for consistent latency
model.eval()
with torch.no_grad():
    output = model(tokens)
```

**Training Stability**:

```python
# Gradually enable components
# Phase 1: Train base model
config.use_dag_planner = False
config.use_hierarchical_memory = False
config.use_imagination = False

# Phase 2: Add memory
config.use_hierarchical_memory = True

# Phase 3: Add planning
config.use_dag_planner = True

# Phase 4: Add imagination
config.use_imagination = True
```

### Debugging Tips

**Check Component Status**:

```python
print(f"DAG Planner: {model.use_dag_planner}")
print(f"Hierarchical Memory: {model.use_hierarchical_memory}")
print(f"Imagination: {model.use_imagination}")
```

**Monitor Memory Growth**:

```python
if model.use_hierarchical_memory:
    sizes = model.hierarchical_memory.get_size()
    print(f"Memory: L1={sizes['l1']}, L2={sizes['l2']}, L3={sizes['l3']}")
```

**Visualize DAG Structure**:

```python
if model.use_dag_planner:
    dag = model.dag_planner(x, deterministic=True)
    adj = dag['adj_matrix'][0]
    mask = dag['node_mask'][0]

    # Simple text visualization
    for i in torch.where(mask)[0]:
        deps = torch.where(adj[i] > 0)[0]
        print(f"Node {i}: deps={deps.tolist()}")
```

**Profile Imagination**:

```python
if model.use_imagination:
    import time
    start = time.time()
    alts = model.imagination(x, n_alternatives=4)
    elapsed = time.time() - start
    print(f"Imagination time: {elapsed:.3f}s")
    print(f"Scores: {alts['scores'][0].tolist()}")
```

### Common Pitfalls

**1. Forgetting Deterministic Mode**:

```python
# ✗ Bad: Non-deterministic during eval
model.eval()
dag = model.dag_planner(x)  # Uses stochastic sampling

# ✓ Good: Deterministic during eval
model.eval()
dag = model.dag_planner(x, deterministic=True)
```

**2. Not Clearing Memory Between Episodes**:

```python
# ✗ Bad: Memory accumulates across unrelated tasks
for episode in episodes:
    result = model(episode)  # Memory from previous episodes leaks

# ✓ Good: Clear between episodes
for episode in episodes:
    model.hierarchical_memory.clear()
    result = model(episode)
```

**3. Ignoring Execution Order**:

```python
# ✗ Bad: Using DAG nodes in arbitrary order
dag = model.dag_planner(x)
nodes = dag['node_embeddings'][0]
# Process nodes[0], nodes[1], ... (wrong order!)

# ✓ Good: Respect topological order
dag = model.dag_planner(x)
adj = dag['adj_matrix'][0]
mask = dag['node_mask'][0]
order = model.dag_planner.get_execution_order(adj, mask)
for node_id in order:
    process(dag['node_embeddings'][0][node_id])
```

**4. Overloading with All Components**:

```python
# ✗ Bad: Enabling everything without need
config.use_dag_planner = True
config.use_hierarchical_memory = True
config.use_imagination = True
# Result: Slow, high memory, may not improve accuracy

# ✓ Good: Enable based on task requirements
# For math: use_dag_planner=True, others=False
# For long context: use_hierarchical_memory=True, others=False
```

### Hyperparameter Tuning

**DAG Planner**:
- `dag_max_nodes`: Start with 16, increase if tasks require more subtasks
- `dag_min_nodes`: Set to 2-3 for simple tasks, 5+ for complex tasks
- `manifold_dim`: 64 is generally sufficient, try 128 for very complex reasoning

**Hierarchical Memory**:
- `l1_capacity`: 64-128 for working memory (recent tokens)
- `l2_capacity`: 512-1024 for short-term (recent examples)
- `l3_capacity`: 2048-8192 for long-term (consolidated knowledge)
- `threshold`: 0.3 is standard, lower (0.2) for more recall, higher (0.4) for precision

**Imagination**:
- `imagination_steps`: 4 is standard, 2 for speed, 8 for quality
- `imagination_n_alternatives`: 4 is standard, 2 for speed, 8+ for exploration
- `noise_scale`: 1.0 is standard, lower (0.5) for conservative exploration

---

## References

### Related Documentation

- [Model Architecture Overview](../README.md)
- [Training Guide](../examples/README.md)
- [Configuration Reference](../../neuromanifold_gpt/config/base.py)

### Example Scripts

- **DAG Planner**: [`docs/examples/dag_planner_usage.py`](../examples/dag_planner_usage.py)
- **Hierarchical Memory**: [`docs/examples/hierarchical_memory_usage.py`](../examples/hierarchical_memory_usage.py)
- **Imagination**: [`docs/examples/imagination_usage.py`](../examples/imagination_usage.py)

### Source Code

- **ForcedDAGPlanner**: [`neuromanifold_gpt/model/planning/dag_planner.py`](../../neuromanifold_gpt/model/planning/dag_planner.py)
- **HierarchicalEngramMemory**: [`neuromanifold_gpt/model/memory/hierarchical_engram.py`](../../neuromanifold_gpt/model/memory/hierarchical_engram.py)
- **ConsistencyImaginationModule**: [`neuromanifold_gpt/model/imagination.py`](../../neuromanifold_gpt/model/imagination.py)
- **Model Integration**: [`neuromanifold_gpt/model/gpt.py`](../../neuromanifold_gpt/model/gpt.py)

### Academic Background

**System 1 vs System 2 Thinking**:
- Kahneman, D. (2011). *Thinking, Fast and Slow*

**DAG-based Planning**:
- Silver et al. (2017). *Mastering the game of Go without human knowledge*
- Xie et al. (2021). *Planning with Learned Object Importance*

**Memory Consolidation**:
- McClelland et al. (1995). *Why there are complementary learning systems in the hippocampus and neocortex*
- Kumaran et al. (2016). *What Learning Systems do Intelligent Agents Need?*

**Consistency Models**:
- Song et al. (2023). *Consistency Models* (arXiv:2303.01469)
- Ho et al. (2020). *Denoising Diffusion Probabilistic Models*

---

## Changelog

### Version 1.0.0 (2026-01-15)
- Initial API documentation
- Complete reference for all three System 2 reasoning components
- Integration guide and configuration examples
- Best practices and troubleshooting guide

---

**Maintained by**: NeuroManifoldGPT Team
**Last Updated**: 2026-01-15
**Version**: 1.0.0
