"""
vektor.hnsw.algorithms
-----------------------
The five algorithms from Malkov & Yashunin (2018).

Naming follows the paper exactly:
  Algorithm 1 — INSERT
  Algorithm 2 — SEARCH-LAYER
  Algorithm 3 — SELECT-NEIGHBORS-SIMPLE
  Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC
  Algorithm 5 — K-NN-SEARCH

All distance comparisons use vektor.distance.compute_distance via the
graph's distance function — never raw NumPy operations directly here.

Layer indexing is 0-based throughout. The paper uses 1-based indexing
in pseudocode. Every paper line that says "layer lc" means layer lc-1
in this implementation.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------
# NodeID is a slot index (int) matching the VectorStore's slot system.
# Graph is: dict[NodeID, dict[layer: int, list[NodeID]]]
# DistFn is: Callable[[NodeID, NodeID], float]
#   — a closure that computes distance between two node slot IDs
#   — higher = better for cosine/dot, lower = better for euclidean
# For HNSW internally we always work in DISTANCE (lower = closer).
# The DistFn must return a value where lower means more similar,
# regardless of the collection metric.

NodeID = int
Graph = dict[NodeID, dict[int, list[NodeID]]]
DistFn = Callable[[NodeID, NodeID], float]


# ---------------------------------------------------------------------------
# Algorithm 2 — SEARCH-LAYER
# ---------------------------------------------------------------------------

def search_layer(
    query_vector: np.ndarray,
    entry_points: list[NodeID],
    ef: int,
    layer: int,
    graph: Graph,
    vectors: dict[NodeID, np.ndarray],
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    skip_ids: Optional[set[NodeID]] = None,
) -> list[tuple[float, NodeID]]:
    """
    Algorithm 2 from the paper. Core beam search at a single graph layer.

    Explores the graph greedily from entry_points, maintains a candidate
    heap and a result heap, and terminates when no further improvement
    is possible.

    Args:
        query_vector:  The query as a float32 ndarray.
        entry_points:  Starting node IDs for this layer's search.
        ef:            Beam width — maximum result set size.
        layer:         The graph layer to search (0-indexed).
        graph:         The full adjacency structure.
        vectors:       NodeID → float32 ndarray mapping.
        dist_fn:       Function(ndarray, ndarray) → float.
                       Must return lower values for more similar vectors.
        skip_ids:      Optional set of NodeIDs to treat as non-existent.
                       Used in Phase 9 for tombstone-aware search.
                       Pass None or empty set in Phase 7.

    Returns:
        List of (distance, node_id) tuples, sorted ascending by distance.
        Length <= ef.
    """
    if skip_ids is None:
        skip_ids = set()

    visited: set[NodeID] = set()

    # candidates: min-heap of (distance, node_id) — closest at top
    candidates: list[tuple[float, NodeID]] = []

    # W: max-heap of (distance, node_id) — farthest at top
    # Python heapq is min-heap → store (-distance, node_id)
    W: list[tuple[float, NodeID]] = []

    for ep in entry_points:
        if ep in skip_ids:
            continue
        d = dist_fn(query_vector, vectors[ep])
        heapq.heappush(candidates, (d, ep))
        heapq.heappush(W, (-d, ep))  # negated for max-heap
        visited.add(ep)

    while candidates:
        # Closest unexplored candidate
        c_dist, c_id = heapq.heappop(candidates)

        # Farthest current result (negate back to get actual distance)
        f_dist = -W[0][0]

        # Termination: if the closest candidate is farther than our worst
        # result, no further exploration can improve W.
        if c_dist > f_dist:
            break

        # Expand c_id's neighbours at this layer
        neighbours = graph.get(c_id, {}).get(layer, [])
        for neighbour_id in neighbours:
            if neighbour_id in visited or neighbour_id in skip_ids:
                continue
            visited.add(neighbour_id)

            n_dist = dist_fn(query_vector, vectors[neighbour_id])
            f_dist = -W[0][0]

            # Accept this neighbour if the result set isn't full yet,
            # or if it's closer than the current worst result.
            if len(W) < ef or n_dist < f_dist:
                heapq.heappush(candidates, (n_dist, neighbour_id))
                heapq.heappush(W, (-n_dist, neighbour_id))

                # Prune W to ef size
                if len(W) > ef:
                    farthest = max(W, key=lambda x: -x[0])
                    W.remove(farthest)
                    heapq.heapify(W) # removes the farthest (largest negated = smallest real)

    # Return results sorted ascending by distance
    results = [(-neg_d, nid) for neg_d, nid in W]
    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Algorithm 3 — SELECT-NEIGHBORS-SIMPLE
# ---------------------------------------------------------------------------

def select_neighbors_simple(
    candidates: list[tuple[float, NodeID]],
    M: int,
) -> list[NodeID]:
    """
    Algorithm 3. Return the M closest nodes from candidates.

    Args:
        candidates: List of (distance, node_id) tuples.
        M:          Maximum number of neighbours to select.

    Returns:
        List of NodeIDs, up to M, sorted ascending by distance.
    """
    sorted_candidates = sorted(candidates, key=lambda x: x[0])
    return [nid for _, nid in sorted_candidates[:M]]


# ---------------------------------------------------------------------------
# Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC
# ---------------------------------------------------------------------------

def select_neighbors_heuristic(
    query_vector: np.ndarray,
    candidates: list[tuple[float, NodeID]],
    M: int,
    vectors: dict[NodeID, np.ndarray],
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    layer: int,
    extend_candidates: bool = False,
    keep_pruned_connections: bool = False,
    graph: Optional[Graph] = None,
) -> list[NodeID]:
    """
    Algorithm 4. Heuristic neighbour selector that promotes graph diversity.

    Unlike Algorithm 3, this does not simply return the M closest candidates.
    It accepts a candidate only if it is closer to the query than it is to
    any already-accepted neighbour. This prevents locally dense clusters from
    monopolising the neighbour list.

    [Certain] This is what separates a correct HNSW implementation from a
    mediocre one. Replacing this with Algorithm 3 produces 5-15% lower recall.

    Core loop logic:
        For each candidate c (nearest-first):
            accepted = False
            For each already-selected neighbour r:
                if dist(c, r) < dist(query, c):
                    # c is closer to an existing neighbour than to the query
                    # → accepting c would create redundant local coverage
                    accepted = False; break
            If no existing neighbour is closer to c than the query is:
                → accept c

    Args:
        query_vector:           The new node's vector.
        candidates:             (distance, node_id) list from SEARCH-LAYER.
        M:                      Max neighbours to select.
        vectors:                NodeID → vector mapping.
        dist_fn:                Distance function (lower = more similar).
        layer:                  Current layer (affects Mmax at layer 0).
        extend_candidates:      If True, add neighbours of candidates to pool.
                                Improves recall on sparse datasets. Default False.
        keep_pruned_connections: If True, fill remaining slots with rejected
                                 candidates when fewer than M pass the heuristic.
                                 Default False.
        graph:                  Required if extend_candidates=True.

    Returns:
        List of NodeIDs, up to M.
    """
    # Work with a mutable heap copy sorted by distance
    W: list[tuple[float, NodeID]] = sorted(candidates, key=lambda x: x[0])

    # Optional: extend candidate pool with neighbours of candidates
    if extend_candidates and graph is not None:
        existing_ids = {nid for _, nid in W}
        extensions = []
        for _, nid in W:
            for neighbour in graph.get(nid, {}).get(layer, []):
                if neighbour not in existing_ids:
                    d = dist_fn(query_vector, vectors[neighbour])
                    extensions.append((d, neighbour))
                    existing_ids.add(neighbour)
        W = sorted(W + extensions, key=lambda x: x[0])

    R: list[tuple[float, NodeID]] = []    # accepted neighbours
    W_discarded: list[tuple[float, NodeID]] = []  # rejected candidates

    for e_dist, e_id in W:
        if len(R) >= M:
            break

        # Check if e is closer to query than to any already-selected neighbour
        e_vector = vectors[e_id]
        closer_to_existing = False

        for r_dist, r_id in R:
            dist_e_to_r = dist_fn(e_vector, vectors[r_id])
            if dist_e_to_r < e_dist:
                # e is closer to r than to query → redundant, skip
                closer_to_existing = True
                break

        if not closer_to_existing:
            R.append((e_dist, e_id))
        else:
            W_discarded.append((e_dist, e_id))

    # Optional: fill remaining slots with best rejected candidates
    if keep_pruned_connections and len(R) < M:
        for d, nid in W_discarded:
            if len(R) >= M:
                break
            R.append((d, nid))

    return [nid for _, nid in R]


# ---------------------------------------------------------------------------
# Algorithm 1 — INSERT
# ---------------------------------------------------------------------------

def insert(
    new_id: NodeID,
    new_vector: np.ndarray,
    graph: Graph,
    vectors: dict[NodeID, np.ndarray],
    entry_point: Optional[NodeID],
    max_layer: int,        # current top layer of the graph
    new_node_layer: int,   # assigned layer for the new node
    M: int,
    ef_construction: int,
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    use_heuristic: bool = True,
) -> tuple[NodeID, int]:
    """
    Algorithm 1. Insert a new node into the HNSW graph.

    Modifies graph in-place. Returns updated (entry_point, max_layer).

    Args:
        new_id:          Slot ID of the new node.
        new_vector:      Float32 vector of the new node.
        graph:           Adjacency structure (modified in-place).
        vectors:         NodeID → vector mapping (new_id must already be here).
        entry_point:     Current global entry point. None if graph is empty.
        max_layer:       Current highest layer in the graph.
        new_node_layer:  The assigned layer for new_id.
        M:               Maximum connections per layer (2M at layer 0).
        ef_construction: Beam width during construction search.
        dist_fn:         Distance function (lower = more similar).
        use_heuristic:   If True, use Algorithm 4. If False, use Algorithm 3.
                         Always use True in production. False only for Step 5 testing.

    Returns:
        (new_entry_point, new_max_layer)
    """
    # First insertion — no graph exists yet
    if entry_point is None:
        graph[new_id] = {lc: [] for lc in range(new_node_layer + 1)}
        return new_id, new_node_layer

    # Initialise the new node's adjacency lists for all its layers
    graph[new_id] = {lc: [] for lc in range(new_node_layer + 1)}

    ep = [entry_point]  # current entry points, mutated as we descend

    # Phase 1: Descend from max_layer to new_node_layer+1
    # Use ef=1 — we only want to find the closest entry point, not explore broadly
    for lc in range(max_layer, new_node_layer, -1):
        results = search_layer(
            query_vector=new_vector,
            entry_points=ep,
            ef=1,
            layer=lc,
            graph=graph,
            vectors=vectors,
            dist_fn=dist_fn,
        )
        ep = [results[0][1]] if results else ep

    # Phase 2: Insert at layers new_node_layer down to 0
    for lc in range(min(max_layer, new_node_layer), -1, -1):
        Mmax = 2 * M if lc == 0 else M

        candidates = search_layer(
            query_vector=new_vector,
            entry_points=ep,
            ef=ef_construction,
            layer=lc,
            graph=graph,
            vectors=vectors,
            dist_fn=dist_fn,
        )

        # Select neighbours for the new node at this layer
        target_M = 2 * M if lc == 0 else M

        if use_heuristic:
            neighbours = select_neighbors_heuristic(
            query_vector=new_vector,
            candidates=candidates,
            M=target_M,
            vectors=vectors,
            dist_fn=dist_fn,
            layer=lc,
            graph=graph,
            )
        else:
            neighbours = select_neighbors_simple(
            candidates,
            target_M,
            )
        
        # Connect new node to its selected neighbours
        graph[new_id][lc] = neighbours
        graph[new_id][lc] = list(dict.fromkeys(graph[new_id][lc]))  # remove duplicates

        # Enforce bidirectional edges + shrink if needed
        for neighbour_id in neighbours:
            if lc not in graph[neighbour_id]:
                graph[neighbour_id][lc] = []

            # Add new_id to neighbour's list
            if new_id != neighbour_id:

                if new_id not in graph[neighbour_id][lc]:
                    graph[neighbour_id][lc].append(new_id)
            
            # Shrink if neighbour now exceeds Mmax
            if len(graph[neighbour_id][lc]) > Mmax:

                neighbour_vector = vectors[neighbour_id]

                existing_candidates = [
                    (dist_fn(neighbour_vector, vectors[nid]), nid)
                    for nid in graph[neighbour_id][lc]
                ]

                old_neighbours = set(graph[neighbour_id][lc])

                if use_heuristic:
                    trimmed = select_neighbors_heuristic(
                        query_vector=neighbour_vector,
                        candidates=existing_candidates,
                        M=Mmax,
                        vectors=vectors,
                        dist_fn=dist_fn,
                        layer=lc,
                        graph=graph,
                    )
                else:
                    trimmed = select_neighbors_simple(existing_candidates, Mmax)

                graph[neighbour_id][lc] = trimmed

                new_neighbours = set(trimmed)

                removed = old_neighbours - new_neighbours

                for removed_id in removed:

                    if lc not in graph.get(removed_id, {}):
                        continue

                    if neighbour_id in graph[removed_id][lc]:
                        graph[removed_id][lc].remove(neighbour_id)

                # Ensure surviving neighbours remain symmetric
                for kept_id in new_neighbours:

                    if lc not in graph[kept_id]:
                        graph[kept_id][lc] = []

                    if neighbour_id not in graph[kept_id][lc]:
                        graph[kept_id][lc].append(neighbour_id)

        # Update entry points for next layer descent
        ep = [res[1] for res in candidates[:ef_construction]]

    # Update global entry point if new node reaches a higher layer
    if new_node_layer > max_layer:
        return new_id, new_node_layer

    return entry_point, max_layer


# ---------------------------------------------------------------------------
# Algorithm 5 — K-NN-SEARCH
# ---------------------------------------------------------------------------

def knn_search(
    query_vector: np.ndarray,
    k: int,
    ef: int,
    entry_point: NodeID,
    max_layer: int,
    graph: Graph,
    vectors: dict[NodeID, np.ndarray],
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    skip_ids: Optional[set[NodeID]] = None,
) -> list[tuple[float, NodeID]]:
    """
    Algorithm 5. K-nearest-neighbour search across all layers.

    Args:
        query_vector: Float32 query vector.
        k:            Number of nearest neighbours to return.
        ef:           Beam width at layer 0. Must be >= k.
        entry_point:  Global entry point (highest-layer node).
        max_layer:    Highest layer in the graph.
        graph:        Full adjacency structure.
        vectors:      NodeID → float32 vector mapping.
        dist_fn:      Distance function (lower = more similar).
        skip_ids:     Optional tombstoned node IDs to exclude from results.

    Returns:
        List of (distance, node_id) tuples, length <= k, sorted ascending.

    Raises:
        InvalidEFError: ef < k.
    """
    from vektor.hnsw.exceptions import InvalidEFError
    if ef < k:
        raise InvalidEFError(
            f"ef ({ef}) must be >= k ({k}). "
            f"Increase ef or decrease k."
        )

    if skip_ids is None:
        skip_ids = set()

    ep = [entry_point]

    # Descend from max_layer to layer 1 using ef=1 (greedy navigation)
    for lc in range(max_layer, 0, -1):
        results = search_layer(
            query_vector=query_vector,
            entry_points=ep,
            ef=1,
            layer=lc,
            graph=graph,
            vectors=vectors,
            dist_fn=dist_fn,
            skip_ids=skip_ids,
        )
        ep = [results[0][1]] if results else ep

    # Search layer 0 with full ef
    results = search_layer(
        query_vector=query_vector,
        entry_points=ep,
        ef=ef,
        layer=0,
        graph=graph,
        vectors=vectors,
        dist_fn=dist_fn,
        skip_ids=skip_ids,
    )

    # Filter skip_ids from final results and return top k
    filtered = [(d, nid) for d, nid in results if nid not in skip_ids]
    return filtered[:k]