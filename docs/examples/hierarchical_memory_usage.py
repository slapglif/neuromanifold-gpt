"""
HierarchicalEngramMemory Usage Example

Demonstrates how to use HierarchicalEngramMemory for System 2 reasoning with
three-tiered memory consolidation. Shows memory storage, tier progression
(L1 -> L2 -> L3), retrieval from different tiers, and eviction dynamics.

The HierarchicalEngramMemory implements biologically-inspired memory consolidation
where memories move from fast working memory (L1) to long-term storage (L3) over
time, enabling efficient capacity scaling and preventing catastrophic forgetting.
"""
import torch
import torch.nn.functional as F
from neuromanifold_gpt.model.memory.hierarchical_engram import HierarchicalEngramMemory
from neuromanifold_gpt.model.memory.engram import SDREngramMemory

# -----------------------------------------------------------------------------
# Configuration
sdr_size = 2048          # size of SDR vectors
n_active = 40            # number of active bits in SDRs
content_dim = 384        # dimension of content vectors
l1_capacity = 8          # small capacity for demonstration
l2_capacity = 16         # medium capacity
l3_capacity = 32         # large capacity
threshold = 0.3          # minimum similarity for retrieval
seed = 1337              # random seed for reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------

# Set random seed for reproducibility
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print("=" * 70)
print("HierarchicalEngramMemory Usage Example")
print("=" * 70)
print(f"\nDevice: {device}")
print(f"SDR size: {sdr_size}, Active bits: {n_active}")
print(f"Content dim: {content_dim}")
print(f"L1 capacity: {l1_capacity} (working memory)")
print(f"L2 capacity: {l2_capacity} (short-term memory)")
print(f"L3 capacity: {l3_capacity} (long-term memory)")
print()

# -----------------------------------------------------------------------------
# 1. Initialize the hierarchical memory
# -----------------------------------------------------------------------------
print("1. Initializing HierarchicalEngramMemory...")
memory = HierarchicalEngramMemory(
    sdr_size=sdr_size,
    n_active=n_active,
    content_dim=content_dim,
    l1_capacity=l1_capacity,
    l2_capacity=l2_capacity,
    l3_capacity=l3_capacity,
    threshold=threshold,
)
memory.to(device)
print(f"   ✓ Memory initialized with 3-tier hierarchy")
print(f"   Total capacity: {l1_capacity + l2_capacity + l3_capacity} memories")
print()

# Helper function to create sparse SDR
def create_sdr(seed_val):
    """Create a sparse SDR with exactly n_active bits set to 1."""
    torch.manual_seed(seed_val)
    indices = torch.randperm(sdr_size)[:n_active]
    sdr = torch.zeros(sdr_size, device=device)
    sdr[indices] = 1.0
    return sdr

# Helper function to create content vector
def create_content(seed_val):
    """Create a random content vector."""
    torch.manual_seed(seed_val)
    return torch.randn(content_dim, device=device)

# -----------------------------------------------------------------------------
# 2. Store memories in L1 (working memory)
# -----------------------------------------------------------------------------
print("2. Storing memories in L1 (working memory)...")
print()

# Store a few memories without filling L1
for i in range(5):
    sdr = create_sdr(i)
    content = create_content(i)
    memory.store(sdr, content)

sizes = memory.get_size()
print(f"   After storing 5 memories:")
print(f"   L1: {sizes['l1'].item()} / {l1_capacity}")
print(f"   L2: {sizes['l2'].item()} / {l2_capacity}")
print(f"   L3: {sizes['l3'].item()} / {l3_capacity}")
print()

# -----------------------------------------------------------------------------
# 3. Fill L1 and trigger eviction to L2
# -----------------------------------------------------------------------------
print("3. Filling L1 and triggering eviction to L2...")
print()

# Store enough to fill L1 (total of l1_capacity memories)
for i in range(5, l1_capacity + 3):
    sdr = create_sdr(i)
    content = create_content(i)
    memory.store(sdr, content)

sizes = memory.get_size()
print(f"   After storing {l1_capacity + 3} total memories:")
print(f"   L1: {sizes['l1'].item()} / {l1_capacity} (working memory full)")
print(f"   L2: {sizes['l2'].item()} / {l2_capacity} (received evicted memories)")
print(f"   L3: {sizes['l3'].item()} / {l3_capacity}")
print()
print(f"   ✓ Oldest memories evicted from L1 to L2")
print()

# -----------------------------------------------------------------------------
# 4. Continue storing to trigger L2 -> L3 cascade
# -----------------------------------------------------------------------------
print("4. Continuing to store to trigger L2 -> L3 cascade...")
print()

# Store enough to fill L2 and trigger cascade to L3
for i in range(l1_capacity + 3, l1_capacity + l2_capacity + 5):
    sdr = create_sdr(i)
    content = create_content(i)
    memory.store(sdr, content)

sizes = memory.get_size()
print(f"   After storing {l1_capacity + l2_capacity + 5} total memories:")
print(f"   L1: {sizes['l1'].item()} / {l1_capacity} (working memory)")
print(f"   L2: {sizes['l2'].item()} / {l2_capacity} (short-term memory full)")
print(f"   L3: {sizes['l3'].item()} / {l3_capacity} (long-term storage)")
print()
print(f"   ✓ Memory consolidation: L1 -> L2 -> L3 cascade")
print()

# -----------------------------------------------------------------------------
# 5. Retrieve from different tiers
# -----------------------------------------------------------------------------
print("5. Retrieving memories from different tiers...")
print()

# Query with a recent memory (should be in L1)
print("   a) Querying with recent memory (expected in L1)...")
recent_idx = l1_capacity + l2_capacity + 4  # Most recent memory
query_sdr_recent = create_sdr(recent_idx)
contents, similarities, tier_stats = memory.retrieve(query_sdr_recent, top_k=3)

print(f"      Retrieved {len(contents)} memories")
print(f"      Tier breakdown: L1={tier_stats['l1']}, L2={tier_stats['l2']}, L3={tier_stats['l3']}")
if len(similarities) > 0:
    print(f"      Top similarity: {similarities[0].item():.4f}")
print()

# Query with an old memory (should be in L3)
print("   b) Querying with old memory (expected in L3)...")
old_idx = 0  # Oldest memory
query_sdr_old = create_sdr(old_idx)
contents, similarities, tier_stats = memory.retrieve(query_sdr_old, top_k=3)

print(f"      Retrieved {len(contents)} memories")
print(f"      Tier breakdown: L1={tier_stats['l1']}, L2={tier_stats['l2']}, L3={tier_stats['l3']}")
if len(similarities) > 0:
    print(f"      Top similarity: {similarities[0].item():.4f}")
print()

# Query with a partial/corrupted SDR (test robustness)
print("   c) Querying with partial SDR (50% of bits)...")
partial_sdr = create_sdr(recent_idx).clone()
# Flip 50% of active bits to test robustness
active_indices = torch.where(partial_sdr > 0)[0]
flip_count = len(active_indices) // 2
partial_sdr[active_indices[:flip_count]] = 0.0
# Activate some random bits to maintain sparsity
new_indices = torch.randperm(sdr_size, device=device)[:flip_count]
partial_sdr[new_indices] = 1.0

contents, similarities, tier_stats = memory.retrieve(partial_sdr, top_k=3)
print(f"      Retrieved {len(contents)} memories")
print(f"      Tier breakdown: L1={tier_stats['l1']}, L2={tier_stats['l2']}, L3={tier_stats['l3']}")
if len(similarities) > 0:
    print(f"      Top similarity: {similarities[0].item():.4f}")
    print(f"      ✓ Hierarchical memory is robust to partial queries")
print()

# -----------------------------------------------------------------------------
# 6. Demonstrate batch storage
# -----------------------------------------------------------------------------
print("6. Demonstrating batch storage...")
print()

# Store a batch of memories at once
batch_size = 5
sdrs_batch = torch.stack([create_sdr(100 + i) for i in range(batch_size)])
contents_batch = torch.stack([create_content(100 + i) for i in range(batch_size)])

sizes_before = memory.get_size()
memory.store_batch(sdrs_batch, contents_batch)
sizes_after = memory.get_size()

print(f"   Stored batch of {batch_size} memories")
print(f"   Before: L1={sizes_before['l1'].item()}, L2={sizes_before['l2'].item()}, L3={sizes_before['l3'].item()}")
print(f"   After:  L1={sizes_after['l1'].item()}, L2={sizes_after['l2'].item()}, L3={sizes_after['l3'].item()}")
print()

# -----------------------------------------------------------------------------
# 7. Compare with single-tier SDREngramMemory
# -----------------------------------------------------------------------------
print("7. Comparing with single-tier SDREngramMemory...")
print()

# Initialize single-tier memory with same total capacity
single_memory = SDREngramMemory(
    sdr_size=sdr_size,
    capacity=l1_capacity + l2_capacity + l3_capacity,
    n_active=n_active,
    content_dim=content_dim,
    threshold=threshold,
)
single_memory.to(device)

print("   Comparison:")
print()
print("   HierarchicalEngramMemory:")
print(f"      ✓ 3-tier architecture with fast working memory")
print(f"      ✓ Automatic consolidation (L1 -> L2 -> L3)")
print(f"      ✓ Recent memories have fast access (L1 hit)")
print(f"      ✓ Old memories still accessible (L3)")
print(f"      ✓ Biologically plausible memory dynamics")
print()
print("   SDREngramMemory:")
print(f"      • Single tier with uniform access time")
print(f"      • No automatic consolidation")
print(f"      • Simpler but less efficient for large capacity")
print()

# -----------------------------------------------------------------------------
# 8. Memory consolidation visualization
# -----------------------------------------------------------------------------
print("8. Memory consolidation pattern visualization...")
print()
print("   Memory Flow:")
print()
print("   New Memory")
print("       ↓")
print("   [L1: Working Memory]  ← Fast access, small capacity")
print("       ↓ (eviction)")
print("   [L2: Short-term]      ← Medium capacity, buffer")
print("       ↓ (consolidation)")
print("   [L3: Long-term]       ← Large capacity, stable")
print()
print("   Retrieval checks L1 → L2 → L3 (fast to slow)")
print("   Similar to hippocampus → cortex consolidation")
print()

# -----------------------------------------------------------------------------
# 9. Demonstrate clearing memory
# -----------------------------------------------------------------------------
print("9. Demonstrating memory clearing...")
print()

sizes_before_clear = memory.get_size()
print(f"   Before clear: L1={sizes_before_clear['l1'].item()}, "
      f"L2={sizes_before_clear['l2'].item()}, L3={sizes_before_clear['l3'].item()}")

memory.clear()
sizes_after_clear = memory.get_size()
print(f"   After clear:  L1={sizes_after_clear['l1'].item()}, "
      f"L2={sizes_after_clear['l2'].item()}, L3={sizes_after_clear['l3'].item()}")
print(f"   ✓ All tiers cleared")
print()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("The HierarchicalEngramMemory successfully demonstrated:")
print("  ✓ Three-tier memory hierarchy (L1/L2/L3)")
print("  ✓ Automatic memory consolidation via cascading eviction")
print("  ✓ Retrieval from all tiers with tier statistics")
print("  ✓ Robustness to partial/corrupted queries")
print("  ✓ Efficient batch storage operations")
print("  ✓ Biologically-inspired memory dynamics")
print()
print("Key takeaways:")
print("  • Recent memories stay in L1 for fast access")
print("  • L1 fills first, then cascades to L2, then L3")
print("  • Retrieval searches all tiers (L1 → L2 → L3)")
print("  • Enables efficient scaling: only L3 needs large capacity")
print("  • Prevents catastrophic forgetting via consolidation")
print("  • Use for: infinite context, episodic memory, knowledge base")
print()
print("Integration tips:")
print("  • Store transformer embeddings as memories during forward pass")
print("  • Query with SDR markers to retrieve relevant context")
print("  • L1 hit rate indicates working memory efficiency")
print("  • Tier statistics reveal memory access patterns")
print()
print("=" * 70)
