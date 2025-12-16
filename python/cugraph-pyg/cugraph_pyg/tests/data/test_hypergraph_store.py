# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cugraph_pyg.utils.imports import import_optional, MissingModule
from cugraph_pyg.data import HypergraphStore

torch = import_optional("torch")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_basic_api(single_pytorch_worker):
    """Test basic hypergraph store operations"""
    # Create a simple hypergraph with 5 nodes and 3 hyperedges
    # Hyperedge 0: nodes [0, 1, 2]
    # Hyperedge 1: nodes [1, 2, 3]
    # Hyperedge 2: nodes [2, 3, 4]
    
    hypergraph_store = HypergraphStore()
    
    # Create hyperedge index in bipartite format
    # [node_indices, hyperedge_indices]
    node_indices = [0, 1, 2, 1, 2, 3, 2, 3, 4]
    hyperedge_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    hyperedge_index = torch.tensor(
        [node_indices, hyperedge_indices],
        dtype=torch.int64,
        device="cuda",
    )
    
    # Store the hyperedge index
    success = hypergraph_store.put_hyperedge_index(
        hyperedge_index,
        ("node", "in", "hyperedge"),
        num_nodes=5,
        num_hyperedges=3,
    )
    assert success
    
    # Retrieve the hyperedge index
    retrieved = hypergraph_store.get_hyperedge_index(("node", "in", "hyperedge"))
    assert retrieved is not None
    assert torch.equal(hyperedge_index, retrieved)
    
    # Check all hyperedge attributes
    attrs = hypergraph_store.get_all_hyperedge_attrs()
    assert len(attrs) == 1
    assert attrs[0] == ("node", "in", "hyperedge")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_list_format(single_pytorch_worker):
    """Test hypergraph store with list-of-lists hyperedge format"""
    hypergraph_store = HypergraphStore()
    
    # Define hyperedges as list of node lists
    hyperedges = [
        [0, 1, 2],  # Hyperedge 0
        [1, 2, 3],  # Hyperedge 1
        [2, 3, 4],  # Hyperedge 2
    ]
    
    # Store the hyperedge index
    success = hypergraph_store.put_hyperedge_index(
        hyperedges,
        ("node", "in", "hyperedge"),
        num_nodes=5,
        num_hyperedges=3,
    )
    assert success
    
    # Retrieve and verify
    retrieved = hypergraph_store.get_hyperedge_index(("node", "in", "hyperedge"))
    assert retrieved is not None


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_bipartite_conversion(single_pytorch_worker):
    """Test conversion of hyperedges to bipartite representation"""
    hypergraph_store = HypergraphStore()
    
    # Define hyperedges as list of node lists
    hyperedges = [
        [0, 1],     # Hyperedge 0 with 2 nodes
        [1, 2, 3],  # Hyperedge 1 with 3 nodes
        [0, 3, 4],  # Hyperedge 2 with 3 nodes
    ]
    
    # Convert to bipartite
    bipartite = hypergraph_store._to_bipartite(hyperedges)
    
    assert bipartite.shape[0] == 2
    assert bipartite.shape[1] == 8  # Total of 2 + 3 + 3 edges in bipartite
    
    # Check that node indices are in first row and hyperedge indices in second row
    node_indices = bipartite[0].tolist()
    hyperedge_indices = bipartite[1].tolist()
    
    # Hyperedge 0 should connect to nodes 0, 1
    assert node_indices.count(0) >= 1
    assert node_indices.count(1) >= 1
    
    # Should have 3 hyperedges
    assert max(hyperedge_indices) == 2


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_statistics(single_pytorch_worker):
    """Test computation of hyperedge statistics"""
    hypergraph_store = HypergraphStore()
    
    # Create hyperedges with varying cardinalities
    node_indices = [0, 1, 1, 2, 3, 2, 3, 4, 5]
    hyperedge_indices = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    hyperedge_index = torch.tensor(
        [node_indices, hyperedge_indices],
        dtype=torch.int64,
        device="cuda",
    )
    
    hypergraph_store.put_hyperedge_index(
        hyperedge_index,
        ("node", "in", "hyperedge"),
        num_nodes=6,
        num_hyperedges=3,
    )
    
    # Compute statistics
    stats = hypergraph_store.compute_hyperedge_statistics(("node", "in", "hyperedge"))
    
    assert stats['num_hyperedges'] == 3
    assert stats['min_cardinality'] == 2  # Hyperedge 0 has 2 nodes
    assert stats['max_cardinality'] == 4  # Hyperedge 2 has 4 nodes
    assert abs(stats['avg_cardinality'] - 3.0) < 0.1  # Average is (2+3+4)/3 = 3


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_remove(single_pytorch_worker):
    """Test removal of hyperedge indices"""
    hypergraph_store = HypergraphStore()
    
    hyperedge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    
    hypergraph_store.put_hyperedge_index(
        hyperedge_index,
        ("node", "in", "hyperedge"),
        num_nodes=3,
        num_hyperedges=1,
    )
    
    # Verify it exists
    assert len(hypergraph_store.get_all_hyperedge_attrs()) == 1
    
    # Remove it
    success = hypergraph_store.remove_hyperedge_index(("node", "in", "hyperedge"))
    assert success
    
    # Verify it's gone
    assert len(hypergraph_store.get_all_hyperedge_attrs()) == 0
    assert hypergraph_store.get_hyperedge_index(("node", "in", "hyperedge")) is None


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_empty_hyperedge(single_pytorch_worker):
    """Test handling of empty hyperedge indices"""
    hypergraph_store = HypergraphStore()
    
    # Try to get non-existent hyperedge
    result = hypergraph_store.get_hyperedge_index(("node", "in", "hyperedge"))
    assert result is None
    
    # Try to remove non-existent hyperedge
    success = hypergraph_store.remove_hyperedge_index(("node", "in", "hyperedge"))
    assert not success


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_multiple_types(single_pytorch_worker):
    """Test storing multiple hyperedge types"""
    hypergraph_store = HypergraphStore()
    
    # Store first hyperedge type
    hyperedge_index1 = torch.tensor(
        [[0, 1, 2], [0, 0, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    hypergraph_store.put_hyperedge_index(
        hyperedge_index1,
        ("author", "writes", "paper"),
        num_nodes=3,
        num_hyperedges=1,
    )
    
    # Store second hyperedge type
    hyperedge_index2 = torch.tensor(
        [[0, 1], [0, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    hypergraph_store.put_hyperedge_index(
        hyperedge_index2,
        ("user", "likes", "item"),
        num_nodes=2,
        num_hyperedges=1,
    )
    
    # Verify both exist
    attrs = hypergraph_store.get_all_hyperedge_attrs()
    assert len(attrs) == 2
    assert ("author", "writes", "paper") in attrs
    assert ("user", "likes", "item") in attrs


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_hypergraph_store_graph_store_access(single_pytorch_worker):
    """Test access to underlying graph store"""
    hypergraph_store = HypergraphStore()
    
    hyperedge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 0]],
        dtype=torch.int64,
        device="cuda",
    )
    
    hypergraph_store.put_hyperedge_index(
        hyperedge_index,
        ("node", "in", "hyperedge"),
        num_nodes=3,
        num_hyperedges=1,
    )
    
    # Access underlying graph store
    graph_store = hypergraph_store.graph_store
    assert graph_store is not None
    
    # Verify the bipartite graph is stored
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) > 0
