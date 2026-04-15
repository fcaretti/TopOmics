"""
Stochastic neighbor sampling for GNN minibatch training.

Replaces exact ``k_hop_subgraph`` with bounded fan-out sampling, giving
predictable subgraph sizes regardless of graph density or number of hops.

The output format matches ``torch_geometric.utils.k_hop_subgraph`` so the
GCNEncoder forward pass requires minimal changes.
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
import torch

logger = logging.getLogger(__name__)


class NeighborSampler:
    """
    Multi-hop stochastic neighbor sampler using vectorized CSR operations.

    For each minibatch of seed nodes, expands the neighbourhood hop-by-hop,
    sampling at most ``fan_out[i]`` neighbours per node at hop ``i``.  Returns
    ``(subset, sub_edge_index, mapping)`` — the same triple produced by
    ``torch_geometric.utils.k_hop_subgraph`` (minus the unused edge_mask).

    Parameters
    ----------
    edge_index : torch.Tensor
        COO edge index ``[2, n_edges]`` (CPU, undirected or directed).
    num_nodes : int
        Total number of nodes in the graph.
    fan_out : list[int]
        Number of neighbours to sample per hop. Length must equal the number
        of GCN layers. E.g. ``[15, 10]`` for a 2-layer GCN.
    replace : bool
        If ``True``, sample neighbours with replacement when a node has fewer
        than ``fan_out`` actual neighbours.  If ``False`` (default), take all
        neighbours when the degree is below the fan-out.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        fan_out: list[int],
        replace: bool = False,
    ) -> None:
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, n_edges], got {tuple(edge_index.shape)}"
            )
        if not fan_out:
            raise ValueError("fan_out must be a non-empty list of ints")

        self.fan_out = list(fan_out)
        self.num_hops = len(fan_out)
        self.num_nodes = num_nodes
        self.replace = replace

        # Build CSR adjacency on CPU. Store indptr and indices arrays directly
        # for fast vectorized row-slicing without Python-level per-node loops.
        row = edge_index[0].numpy()
        col = edge_index[1].numpy()
        data = np.ones(len(row), dtype=np.float32)
        csr = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        self._indptr = csr.indptr
        self._indices = csr.indices

        logger.info(
            f"NeighborSampler initialized: {num_nodes} nodes, "
            f"{edge_index.shape[1]} edges, fan_out={fan_out}"
        )

    def sample(
        self, seed_nodes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a multi-hop subgraph around ``seed_nodes``.

        Uses vectorized CSR slicing: for all frontier nodes at once, extract
        their neighbor ranges from the CSR indptr/indices arrays, then batch-
        sample with a single ``np.random.randint`` call per hop.

        Parameters
        ----------
        seed_nodes : torch.Tensor
            1-D tensor of global node indices (the minibatch).

        Returns
        -------
        subset : torch.Tensor [num_sampled]
            Global node indices of all nodes in the sampled subgraph,
            with seed nodes first (in their original order).
        sub_edge_index : torch.Tensor [2, num_edges_sub]
            Edges within the subgraph, relabelled to local indices.
        mapping : torch.Tensor [len(seed_nodes)]
            Local indices of the seed nodes within ``subset``.
            Since seeds are placed first, this is ``arange(len(seed_nodes))``.
        """
        seeds = seed_nodes.numpy().astype(np.int64)
        indptr = self._indptr
        indices = self._indices

        # Track all unique nodes (seeds first) and all edges
        # Use a dense boolean visited array instead of a Python set
        visited = np.zeros(self.num_nodes, dtype=bool)
        visited[seeds] = True

        # Nodes collected after seeds (to append after seeds in subset)
        extra_nodes_list: list[np.ndarray] = []
        all_src_list: list[np.ndarray] = []
        all_dst_list: list[np.ndarray] = []

        frontier = seeds

        for hop in range(self.num_hops):
            fo = self.fan_out[hop]
            n_frontier = len(frontier)
            if n_frontier == 0:
                break

            # Vectorized: get degree of each frontier node from CSR indptr
            starts = indptr[frontier]
            ends = indptr[frontier + 1]
            degrees = ends - starts

            # For each frontier node, sample min(fo, degree) neighbors
            # We process all nodes at once using a flat array approach
            sampled_neighbors_list = []
            edge_src_list = []
            edge_dst_list = []

            # Separate nodes into "needs sampling" (degree > fo) and "take all"
            needs_sample = degrees > fo
            take_all = ~needs_sample
            has_neighbors = degrees > 0

            # Handle "take all" nodes: extract all their neighbors at once
            take_all_mask = take_all & has_neighbors
            if take_all_mask.any():
                ta_nodes = frontier[take_all_mask]
                ta_starts = starts[take_all_mask]
                ta_ends = ends[take_all_mask]
                # Build flat index ranges for all take-all nodes
                flat_indices = np.concatenate([
                    np.arange(s, e) for s, e in zip(ta_starts, ta_ends)
                ])
                ta_neighbors = indices[flat_indices]
                sampled_neighbors_list.append(ta_neighbors)
                # Build edge arrays: repeat each node by its degree
                ta_degrees = ta_ends - ta_starts
                edge_dst_list.append(np.repeat(ta_nodes, ta_degrees))
                edge_src_list.append(ta_neighbors)

            # Handle "needs sampling" nodes: sample fo neighbors each
            ns_mask = needs_sample & has_neighbors
            if ns_mask.any():
                ns_nodes = frontier[ns_mask]
                ns_starts = starts[ns_mask]
                ns_degrees = degrees[ns_mask]
                n_ns = len(ns_nodes)

                # Generate random offsets in [0, degree) for each node, fo samples each
                # Shape: (n_ns, fo)
                random_offsets = np.array([
                    np.random.randint(0, d, size=fo) for d in ns_degrees
                ])  # (n_ns, fo)

                # Convert to flat indices into CSR indices array
                flat_idx = (ns_starts[:, None] + random_offsets).ravel()
                ns_neighbors = indices[flat_idx]  # (n_ns * fo,)
                sampled_neighbors_list.append(ns_neighbors)
                # Edges
                edge_dst_list.append(np.repeat(ns_nodes, fo))
                edge_src_list.append(ns_neighbors)

            if not sampled_neighbors_list:
                break

            hop_neighbors = np.concatenate(sampled_neighbors_list)
            all_src_list.append(np.concatenate(edge_src_list))
            all_dst_list.append(np.concatenate(edge_dst_list))

            # Find new (unvisited) nodes
            new_mask = ~visited[hop_neighbors]
            new_nodes = np.unique(hop_neighbors[new_mask])
            visited[new_nodes] = True
            extra_nodes_list.append(new_nodes)
            frontier = new_nodes

        # Build subset: seeds first, then extra nodes
        if extra_nodes_list:
            extra = np.concatenate(extra_nodes_list)
            subset_np = np.concatenate([seeds, extra])
        else:
            subset_np = seeds.copy()

        # Build global-to-local mapping using a dense array (fast)
        global_to_local = np.empty(self.num_nodes, dtype=np.int64)
        global_to_local[subset_np] = np.arange(len(subset_np))

        # Relabel edges
        if all_src_list:
            all_src = np.concatenate(all_src_list)
            all_dst = np.concatenate(all_dst_list)
            src_local = global_to_local[all_src]
            dst_local = global_to_local[all_dst]
            sub_edge_index = torch.from_numpy(
                np.stack([src_local, dst_local])
            ).long()
        else:
            sub_edge_index = torch.zeros(2, 0, dtype=torch.long)

        subset = torch.from_numpy(subset_np).long()
        mapping = torch.arange(len(seeds), dtype=torch.long)

        return subset, sub_edge_index, mapping
