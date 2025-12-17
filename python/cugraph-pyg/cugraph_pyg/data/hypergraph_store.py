# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import cupy
import pandas

from cugraph_pyg.utils.imports import import_optional, MissingModule
from cugraph_pyg.data.graph_store import GraphStore
from cugraph_pyg.data.hypergraph_utils import convert_hyperedges_to_bipartite

# Have to use import_optional even though these are required
# dependencies in order to build properly.
torch_geometric = import_optional("torch_geometric")
torch = import_optional("torch")
cudf = import_optional("cudf")

TensorType = Union[
    "torch.Tensor", cupy.ndarray, np.ndarray, "cudf.Series", pandas.Series
]


class HypergraphStore(
    object
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.data.GraphStore
):
    """
    cuGraph-backed PyG HypergraphStore implementation that extends GraphStore
    with hypergraph capabilities. In hypergraphs, edges (hyperedges) can
    connect more than two nodes.
    
    Hypergraphs are represented using a bipartite graph structure where:
    - One set of nodes represents the original graph nodes
    - Another set represents hyperedges
    - Edges connect nodes to the hyperedges they belong to
    
    This allows efficient storage and sampling of hypergraphs using the
    existing cuGraph infrastructure.
    """

    def __init__(self):
        """
        Constructs a new, empty HypergraphStore object.
        """
        self.__hyperedges = {}
        self.__graph_store = GraphStore()
        super().__init__()

    def put_hyperedge_index(
        self,
        hyperedge_index: TensorType,
        edge_type: Tuple[str, str, str],
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
    ) -> bool:
        """
        Store a hyperedge index in the hypergraph store.
        
        Parameters
        ----------
        hyperedge_index : TensorType
            A list or tensor where each element is a list/tensor of node indices
            that form a hyperedge. Can also be a 2D tensor of shape [num_edges, 2]
            where column 0 is node indices and column 1 is hyperedge indices.
        edge_type : Tuple[str, str, str]
            The edge type as (src_type, relation, dst_type).
        num_nodes : int, optional
            The total number of nodes in the graph.
        num_hyperedges : int, optional
            The total number of hyperedges.
            
        Returns
        -------
        bool
            True if successful.
        """
        if isinstance(hyperedge_index, (cupy.ndarray, cudf.Series)):
            hyperedge_index = torch.as_tensor(hyperedge_index, device="cuda")
        elif isinstance(hyperedge_index, np.ndarray):
            hyperedge_index = torch.as_tensor(hyperedge_index, device="cpu")
        elif isinstance(hyperedge_index, pandas.Series):
            hyperedge_index = torch.as_tensor(hyperedge_index.values, device="cpu")
        
        # Store the hyperedge representation
        self.__hyperedges[edge_type] = hyperedge_index
        
        # Convert to bipartite representation and store in underlying GraphStore
        bipartite_edge_index = convert_hyperedges_to_bipartite(hyperedge_index)
        
        # Create edge attribute for the bipartite graph
        if num_nodes is not None and num_hyperedges is not None:
            size = (num_nodes, num_hyperedges)
        else:
            size = None
            
        edge_attr = torch_geometric.data.EdgeAttr(
            edge_type=edge_type,
            layout=torch_geometric.data.graph_store.EdgeLayout.COO,
            is_sorted=False,
            size=size,
        )
        
        # Store in the underlying graph store
        self.__graph_store._put_edge_index(bipartite_edge_index, edge_attr)
        
        return True

    def get_hyperedge_index(
        self, edge_type: Tuple[str, str, str]
    ) -> Optional[TensorType]:
        """
        Retrieve a hyperedge index from the store.
        
        Parameters
        ----------
        edge_type : Tuple[str, str, str]
            The edge type as (src_type, relation, dst_type).
            
        Returns
        -------
        TensorType or None
            The hyperedge index if it exists, None otherwise.
        """
        return self.__hyperedges.get(edge_type)

    def _to_bipartite(
        self,
        hyperedge_index: TensorType,
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
    ) -> "torch.Tensor":
        """
        Convert a hyperedge index to a bipartite graph representation.
        
        DEPRECATED: Use convert_hyperedges_to_bipartite() from hypergraph_utils instead.
        
        In the bipartite representation:
        - Nodes in the original graph form one partition
        - Hyperedges form the other partition
        - An edge exists between node i and hyperedge j if node i is in hyperedge j
        
        Parameters
        ----------
        hyperedge_index : TensorType
            The hyperedge index. If 2D with shape [num_edges, 2], assumes it's
            already in bipartite format (node_idx, hyperedge_idx). Otherwise,
            expects a list-like structure where each element contains node indices
            for that hyperedge.
        num_nodes : int, optional
            The total number of nodes (unused, kept for compatibility).
        num_hyperedges : int, optional
            The total number of hyperedges (unused, kept for compatibility).
            
        Returns
        -------
        torch.Tensor
            A 2D tensor of shape [2, num_edges] representing the bipartite graph
            in COO format. First row is node indices, second row is hyperedge indices.
        """
        return convert_hyperedges_to_bipartite(hyperedge_index)

    def remove_hyperedge_index(self, edge_type: Tuple[str, str, str]) -> bool:
        """
        Remove a hyperedge index from the store.
        
        Parameters
        ----------
        edge_type : Tuple[str, str, str]
            The edge type to remove.
            
        Returns
        -------
        bool
            True if successful, False if the edge type doesn't exist.
        """
        if edge_type not in self.__hyperedges:
            return False
            
        del self.__hyperedges[edge_type]
        
        # Remove from underlying graph store
        edge_attr = torch_geometric.data.EdgeAttr(
            edge_type=edge_type,
            layout=torch_geometric.data.graph_store.EdgeLayout.COO,
        )
        self.__graph_store._remove_edge_index(edge_attr)
        
        return True

    def get_all_hyperedge_attrs(self) -> List[Tuple[str, str, str]]:
        """
        Get all hyperedge types stored in this hypergraph store.
        
        Returns
        -------
        List[Tuple[str, str, str]]
            List of edge types.
        """
        return list(self.__hyperedges.keys())

    @property
    def graph_store(self) -> GraphStore:
        """
        Access the underlying GraphStore for bipartite representation.
        
        Returns
        -------
        GraphStore
            The underlying graph store.
        """
        return self.__graph_store

    @property
    def is_multi_gpu(self) -> bool:
        """
        Check if this is a multi-GPU setup.
        
        Returns
        -------
        bool
            True if running on multiple GPUs.
        """
        return self.__graph_store.is_multi_gpu

    def compute_hyperedge_statistics(
        self, edge_type: Tuple[str, str, str]
    ) -> Dict[str, Union[int, float]]:
        """
        Compute statistics about the hyperedges.
        
        Parameters
        ----------
        edge_type : Tuple[str, str, str]
            The edge type to analyze.
            
        Returns
        -------
        Dict[str, Union[int, float]]
            Dictionary containing:
            - 'num_hyperedges': Total number of hyperedges
            - 'avg_cardinality': Average number of nodes per hyperedge
            - 'max_cardinality': Maximum hyperedge size
            - 'min_cardinality': Minimum hyperedge size
        """
        if edge_type not in self.__hyperedges:
            raise ValueError(f"Edge type {edge_type} not found")
        
        hyperedge_index = self.__hyperedges[edge_type]
        
        # Convert to bipartite format if needed
        if isinstance(hyperedge_index, (list, tuple)):
            hyperedge_index = convert_hyperedges_to_bipartite(hyperedge_index)
        
        # Assuming bipartite format [2, num_edges]
        if isinstance(hyperedge_index, torch.Tensor) and hyperedge_index.dim() == 2:
            if hyperedge_index.shape[0] == 2:
                # Count nodes per hyperedge
                hyperedge_ids = hyperedge_index[1]
                unique_he, counts = torch.unique(hyperedge_ids, return_counts=True)
                
                return {
                    'num_hyperedges': int(len(unique_he)),
                    'avg_cardinality': float(counts.float().mean()),
                    'max_cardinality': int(counts.max()),
                    'min_cardinality': int(counts.min()),
                }
        
        return {
            'num_hyperedges': 0,
            'avg_cardinality': 0.0,
            'max_cardinality': 0,
            'min_cardinality': 0,
        }
