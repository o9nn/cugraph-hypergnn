# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cugraph.datasets import karate
from cugraph_pyg.utils.imports import import_optional, MissingModule

from cugraph_pyg.data import GraphStore

torch = import_optional("torch")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_graph_store_basic_api(single_pytorch_worker):
    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device="cuda")
    dst = torch.as_tensor(df["dst"], device="cuda")

    ei = torch.stack([dst, src])

    num_nodes = karate.number_of_nodes()

    graph_store = GraphStore()
    graph_store.put_edge_index(
        ei, ("person", "knows", "person"), "coo", False, (num_nodes, num_nodes)
    )

    rei = graph_store.get_edge_index(("person", "knows", "person"), "coo")

    assert (ei == rei).all()

    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1

    graph_store.remove_edge_index(("person", "knows", "person"), "coo")
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_graph_store_hyperedge_api(single_pytorch_worker):
    """Test GraphStore hyperedge functionality"""
    graph_store = GraphStore()
    
    # Create a simple hypergraph with 4 nodes and 2 hyperedges
    # Hyperedge 0: connects nodes [0, 1, 2]
    # Hyperedge 1: connects nodes [1, 2, 3]
    
    # Define in bipartite format: [node_indices, hyperedge_indices]
    node_indices = [0, 1, 2, 1, 2, 3]
    hyperedge_indices = [0, 0, 0, 1, 1, 1]
    hyperedge_index = torch.tensor(
        [node_indices, hyperedge_indices],
        dtype=torch.int64,
        device="cuda",
    )
    
    # Store hyperedge index
    success = graph_store.put_hyperedge_index(
        hyperedge_index,
        ("node", "in", "hyperedge"),
        num_nodes=4,
        num_hyperedges=2,
    )
    assert success
    
    # Verify it was stored
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1
    assert edge_attrs[0].edge_type == ("node", "in", "hyperedge")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_graph_store_hyperedge_list_format(single_pytorch_worker):
    """Test GraphStore hyperedge with list-of-lists format"""
    graph_store = GraphStore()
    
    # Define hyperedges as list of node lists
    hyperedges = [
        [0, 1, 2],  # Hyperedge 0
        [1, 2, 3],  # Hyperedge 1
    ]
    
    # Store hyperedge index
    success = graph_store.put_hyperedge_index(
        hyperedges,
        ("node", "in", "hyperedge"),
        num_nodes=4,
        num_hyperedges=2,
    )
    assert success
    
    # Verify it was stored
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1
