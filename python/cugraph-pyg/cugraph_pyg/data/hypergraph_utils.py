# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for hypergraph operations.
"""

from typing import Union, List
import numpy as np

from cugraph_pyg.utils.imports import import_optional

torch = import_optional("torch")


def convert_hyperedges_to_bipartite(
    hyperedge_index: Union[List, "torch.Tensor"],
) -> "torch.Tensor":
    """
    Convert hyperedge representation to bipartite format.
    
    Parameters
    ----------
    hyperedge_index : List or torch.Tensor
        Either:
        - A list of lists where each sublist contains node indices for a hyperedge
        - A 2D tensor in bipartite format [2, num_edges] or [num_edges, 2]
        
    Returns
    -------
    torch.Tensor
        A 2D tensor of shape [2, num_edges] in bipartite COO format.
        Row 0: node indices
        Row 1: hyperedge indices
        
    Examples
    --------
    >>> # List format
    >>> hyperedges = [[0, 1, 2], [1, 2, 3]]
    >>> result = convert_hyperedges_to_bipartite(hyperedges)
    >>> # result shape: [2, 6] (6 edges in bipartite graph)
    
    >>> # Tensor format already in bipartite
    >>> hyperedge_tensor = torch.tensor([[0, 1, 2], [0, 0, 0]])
    >>> result = convert_hyperedges_to_bipartite(hyperedge_tensor)
    >>> # result shape: [2, 3]
    """
    # If already in bipartite format [2, num_edges] or [num_edges, 2]
    if isinstance(hyperedge_index, torch.Tensor):
        if hyperedge_index.dim() == 2:
            if hyperedge_index.shape[0] == 2:
                # Already in [2, num_edges] format
                return hyperedge_index
            elif hyperedge_index.shape[1] == 2:
                # In [num_edges, 2] format, transpose
                return hyperedge_index.t().contiguous()
    
    # Convert list-like hyperedge representation to bipartite
    node_indices = []
    hyperedge_indices = []
    
    if isinstance(hyperedge_index, (list, tuple)):
        for he_idx, nodes in enumerate(hyperedge_index):
            if isinstance(nodes, (torch.Tensor, np.ndarray, list)):
                node_list = torch.as_tensor(nodes).tolist() if not isinstance(nodes, list) else nodes
                for node_idx in node_list:
                    node_indices.append(node_idx)
                    hyperedge_indices.append(he_idx)
            else:
                node_indices.append(int(nodes))
                hyperedge_indices.append(he_idx)
    else:
        # Assume it's a tensor that needs to be interpreted
        # as a flat list of node indices with equal-sized hyperedges
        raise ValueError(
            "hyperedge_index must be either a 2D tensor in bipartite format "
            "or a list of node index lists/tensors for each hyperedge"
        )
    
    # Create the bipartite edge index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    edge_index = torch.tensor(
        [node_indices, hyperedge_indices],
        dtype=torch.int64,
        device=device,
    )
    
    return edge_index
