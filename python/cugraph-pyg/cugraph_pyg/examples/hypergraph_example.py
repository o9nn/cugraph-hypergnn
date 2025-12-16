# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Using HGNN Hypergraph Capabilities in cuGraph-PyG

This example demonstrates how to use the new hypergraph functionality
to work with hypergraphs (graphs where edges can connect more than 2 nodes).
"""

import torch
from cugraph_pyg.data import GraphStore, HypergraphStore

# Example 1: Using GraphStore with hyperedges
print("Example 1: Using GraphStore.put_hyperedge_index()")
print("=" * 60)

# Initialize a GraphStore
graph_store = GraphStore()

# Define a hypergraph with 6 nodes and 3 hyperedges
# Hyperedge 0: connects nodes [0, 1, 2]
# Hyperedge 1: connects nodes [1, 2, 3, 4]
# Hyperedge 2: connects nodes [3, 4, 5]

# Method 1: Using list-of-lists format
hyperedges_list = [
    [0, 1, 2],      # Hyperedge 0
    [1, 2, 3, 4],   # Hyperedge 1
    [3, 4, 5],      # Hyperedge 2
]

graph_store.put_hyperedge_index(
    hyperedges_list,
    ("node", "in", "hyperedge"),
    num_nodes=6,
    num_hyperedges=3,
)

print(f"Stored {len(hyperedges_list)} hyperedges")
print(f"Edge types: {[attr.edge_type for attr in graph_store.get_all_edge_attrs()]}")

# Example 2: Using bipartite tensor format
print("\nExample 2: Using bipartite tensor format")
print("=" * 60)

graph_store2 = GraphStore()

# Define the same hypergraph using bipartite representation
# Format: [node_indices, hyperedge_indices]
node_indices = [0, 1, 2, 1, 2, 3, 4, 3, 4, 5]
hyperedge_indices = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]

hyperedge_tensor = torch.tensor(
    [node_indices, hyperedge_indices],
    dtype=torch.int64,
    device="cuda",
)

graph_store2.put_hyperedge_index(
    hyperedge_tensor,
    ("node", "in", "hyperedge"),
    num_nodes=6,
    num_hyperedges=3,
)

print(f"Stored hypergraph in bipartite format")
print(f"Shape: {hyperedge_tensor.shape}")

# Example 3: Using HypergraphStore for advanced operations
print("\nExample 3: Using HypergraphStore with statistics")
print("=" * 60)

hypergraph_store = HypergraphStore()

# Store hyperedges with varying sizes
hyperedges = [
    [0, 1],           # Small hyperedge (2 nodes)
    [1, 2, 3, 4, 5],  # Large hyperedge (5 nodes)
    [3, 4, 5, 6],     # Medium hyperedge (4 nodes)
]

hypergraph_store.put_hyperedge_index(
    hyperedges,
    ("node", "contains", "hyperedge"),
    num_nodes=7,
    num_hyperedges=3,
)

# Compute hyperedge statistics
stats = hypergraph_store.compute_hyperedge_statistics(
    ("node", "contains", "hyperedge")
)

print(f"Number of hyperedges: {stats['num_hyperedges']}")
print(f"Average hyperedge size: {stats['avg_cardinality']:.2f}")
print(f"Minimum hyperedge size: {stats['min_cardinality']}")
print(f"Maximum hyperedge size: {stats['max_cardinality']}")

# Example 4: Heterogeneous hypergraph
print("\nExample 4: Heterogeneous hypergraph")
print("=" * 60)

hetero_graph_store = GraphStore()

# Store multiple types of hyperedges
# Authors writing papers together
author_hyperedges = [
    [0, 1, 2],  # Paper 0 has authors 0, 1, 2
    [1, 2, 3],  # Paper 1 has authors 1, 2, 3
]

hetero_graph_store.put_hyperedge_index(
    author_hyperedges,
    ("author", "writes", "paper"),
    num_nodes=4,
    num_hyperedges=2,
)

# Users liking items
user_hyperedges = [
    [0, 1, 2, 3],  # Item 0 is liked by users 0, 1, 2, 3
    [2, 3, 4],     # Item 1 is liked by users 2, 3, 4
]

hetero_graph_store.put_hyperedge_index(
    user_hyperedges,
    ("user", "likes", "item"),
    num_nodes=5,
    num_hyperedges=2,
)

print(f"Stored heterogeneous hypergraph with {len(hetero_graph_store.get_all_edge_attrs())} edge types")
for attr in hetero_graph_store.get_all_edge_attrs():
    print(f"  - {attr.edge_type}")

print("\nAll examples completed successfully!")
