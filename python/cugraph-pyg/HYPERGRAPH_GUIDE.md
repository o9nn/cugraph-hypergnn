# Hypergraph Neural Network (HGNN) Support in cuGraph-PyG

## Overview

cuGraph-PyG now supports Hypergraph Neural Networks (HGNNs), extending the traditional GNN graph functionality to handle hypergraphs. In hypergraphs, edges (called hyperedges) can connect more than two nodes, making them ideal for modeling complex relationships in data.

## Key Concepts

### What is a Hypergraph?

A hypergraph is a generalization of a graph where:
- **Nodes**: Same as regular graphs
- **Hyperedges**: Can connect any number of nodes (not just 2)

**Example**: In a research collaboration network:
- Traditional graph: Edge connects 2 authors
- Hypergraph: Hyperedge connects all authors on a paper

### Bipartite Representation

Internally, hypergraphs are stored as bipartite graphs:
- **Partition 1**: Original nodes
- **Partition 2**: Hyperedges
- **Edges**: Connect nodes to hyperedges they belong to

This representation allows efficient storage and sampling using existing cuGraph infrastructure.

## API Reference

### GraphStore.put_hyperedge_index()

Add hypergraph support directly to existing `GraphStore` objects.

```python
def put_hyperedge_index(
    self,
    hyperedge_index: TensorType,
    edge_type: Tuple[str, str, str],
    num_nodes: Optional[int] = None,
    num_hyperedges: Optional[int] = None,
) -> bool
```

**Parameters:**
- `hyperedge_index`: Hyperedge data in one of two formats:
  - **List format**: `[[node1, node2, ...], [node3, node4, ...], ...]`
  - **Bipartite tensor**: `torch.Tensor` of shape `[2, num_edges]`
- `edge_type`: Tuple of `(src_type, relation, dst_type)`
- `num_nodes`: Total number of nodes (optional)
- `num_hyperedges`: Total number of hyperedges (optional)

**Example:**
```python
from cugraph_pyg.data import GraphStore

graph_store = GraphStore()

# Using list format
hyperedges = [
    [0, 1, 2],      # Hyperedge 0
    [1, 2, 3, 4],   # Hyperedge 1
]

graph_store.put_hyperedge_index(
    hyperedges,
    ("node", "in", "hyperedge"),
    num_nodes=5,
    num_hyperedges=2,
)
```

### HypergraphStore

Specialized store for hypergraph operations with additional utilities.

```python
from cugraph_pyg.data import HypergraphStore

hypergraph_store = HypergraphStore()
```

**Key Methods:**

#### put_hyperedge_index()
Store hyperedges in the hypergraph store.

#### get_hyperedge_index()
Retrieve stored hyperedges.

```python
hyperedge_index = hypergraph_store.get_hyperedge_index(
    ("node", "in", "hyperedge")
)
```

#### compute_hyperedge_statistics()
Analyze hyperedge properties.

```python
stats = hypergraph_store.compute_hyperedge_statistics(
    ("node", "in", "hyperedge")
)
# Returns: {
#   'num_hyperedges': int,
#   'avg_cardinality': float,
#   'max_cardinality': int,
#   'min_cardinality': int,
# }
```

#### get_all_hyperedge_attrs()
List all hyperedge types.

#### remove_hyperedge_index()
Remove hyperedges from the store.

## Usage Examples

### Example 1: Simple Hypergraph

```python
import torch
from cugraph_pyg.data import GraphStore

# Create graph store
graph_store = GraphStore()

# Define hyperedges (each list is one hyperedge)
hyperedges = [
    [0, 1, 2],      # 3 nodes in hyperedge 0
    [1, 2, 3, 4],   # 4 nodes in hyperedge 1
    [3, 4, 5],      # 3 nodes in hyperedge 2
]

# Store in graph store
graph_store.put_hyperedge_index(
    hyperedges,
    ("node", "connected_by", "hyperedge"),
    num_nodes=6,
    num_hyperedges=3,
)
```

### Example 2: Bipartite Format

```python
import torch
from cugraph_pyg.data import GraphStore

graph_store = GraphStore()

# Define in bipartite format: [node_idx, hyperedge_idx]
# This represents the same hypergraph as Example 1
node_indices = [0, 1, 2, 1, 2, 3, 4, 3, 4, 5]
hyperedge_indices = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]

hyperedge_tensor = torch.tensor(
    [node_indices, hyperedge_indices],
    dtype=torch.int64,
    device="cuda",
)

graph_store.put_hyperedge_index(
    hyperedge_tensor,
    ("node", "connected_by", "hyperedge"),
    num_nodes=6,
    num_hyperedges=3,
)
```

### Example 3: Heterogeneous Hypergraph

```python
from cugraph_pyg.data import GraphStore

graph_store = GraphStore()

# Authors collaborating on papers
author_collaborations = [
    [0, 1, 2],      # Paper 0
    [1, 2, 3, 4],   # Paper 1
]

graph_store.put_hyperedge_index(
    author_collaborations,
    ("author", "writes", "paper"),
    num_nodes=5,
    num_hyperedges=2,
)

# Users interacting with items
user_interactions = [
    [0, 1, 2],      # Item 0
    [2, 3, 4, 5],   # Item 1
]

graph_store.put_hyperedge_index(
    user_interactions,
    ("user", "interacts", "item"),
    num_nodes=6,
    num_hyperedges=2,
)
```

### Example 4: Statistics and Analysis

```python
from cugraph_pyg.data import HypergraphStore

hypergraph_store = HypergraphStore()

hyperedges = [
    [0, 1],              # Size 2
    [1, 2, 3, 4, 5, 6], # Size 6
    [3, 4],              # Size 2
    [5, 6, 7, 8],        # Size 4
]

hypergraph_store.put_hyperedge_index(
    hyperedges,
    ("node", "in", "hyperedge"),
    num_nodes=9,
    num_hyperedges=4,
)

# Get statistics
stats = hypergraph_store.compute_hyperedge_statistics(
    ("node", "in", "hyperedge")
)

print(f"Total hyperedges: {stats['num_hyperedges']}")
print(f"Avg size: {stats['avg_cardinality']:.2f}")
print(f"Min size: {stats['min_cardinality']}")
print(f"Max size: {stats['max_cardinality']}")
# Output:
# Total hyperedges: 4
# Avg size: 3.50
# Min size: 2
# Max size: 6
```

## Integration with Loaders and Samplers

Hypergraphs stored using `GraphStore.put_hyperedge_index()` can be used with existing cuGraph-PyG loaders:

```python
from cugraph_pyg.data import GraphStore, FeatureStore
from cugraph_pyg.loader import NeighborLoader

# Create stores
graph_store = GraphStore()
feature_store = FeatureStore()

# Add hypergraph data
hyperedges = [[0, 1, 2], [1, 2, 3]]
graph_store.put_hyperedge_index(
    hyperedges,
    ("node", "in", "hyperedge"),
    num_nodes=4,
    num_hyperedges=2,
)

# Add node features
node_features = torch.randn(4, 16)
feature_store.put_tensor(
    node_features,
    group_name="node",
    attr_name="x",
)

# Create loader
loader = NeighborLoader(
    (feature_store, graph_store),
    num_neighbors=[10] * 2,
    batch_size=2,
    input_nodes=("node", torch.tensor([0, 1])),
)

# Use loader for training
for batch in loader:
    # Training code here
    pass
```

## Use Cases

### 1. Research Collaboration Networks
- Nodes: Researchers
- Hyperedges: Papers (connecting all co-authors)

### 2. Social Media Group Interactions
- Nodes: Users
- Hyperedges: Group conversations (connecting all participants)

### 3. Biological Pathways
- Nodes: Proteins/Genes
- Hyperedges: Biological reactions (connecting multiple participants)

### 4. Recommendation Systems
- Nodes: Users or Items
- Hyperedges: User sessions or item bundles

### 5. Knowledge Graphs
- Nodes: Entities
- Hyperedges: N-ary relationships

## Performance Considerations

1. **GPU Acceleration**: All operations are GPU-accelerated using cuGraph and PyTorch
2. **Multi-GPU Support**: Supports distributed hypergraphs across multiple GPUs
3. **Memory Efficiency**: Bipartite representation avoids storing dense incidence matrices
4. **Sampling**: Leverages existing cuGraph sampling infrastructure for fast neighbor sampling

## Migration from Traditional Graphs

Traditional graph operations remain fully compatible:

```python
# Traditional graph edge
graph_store.put_edge_index(
    edge_index,
    ("node", "connects", "node"),
    "coo",
    size=(num_nodes, num_nodes)
)

# Hypergraph edge (3+ nodes)
graph_store.put_hyperedge_index(
    hyperedges,
    ("node", "in", "hyperedge"),
    num_nodes=num_nodes,
    num_hyperedges=num_hyperedges,
)

# Both can coexist in the same graph store
```

## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [cuGraph Documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [Hypergraph Neural Networks (HGNN) Paper](https://arxiv.org/abs/1809.09401)

## Contributing

To contribute hypergraph-related features:

1. Follow the existing code style
2. Add tests to `tests/data/test_hypergraph_store.py`
3. Update documentation
4. Run tests: `pytest -v tests/data/test_hypergraph_store.py`
