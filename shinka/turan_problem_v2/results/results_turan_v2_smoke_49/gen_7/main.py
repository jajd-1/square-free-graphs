# EVOLVE-BLOCK-START
import numpy as np


def construct_new_graph(A, rng=None):
    """Repair and improve the supplied graph into a dense C4-free graph."""
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("A must be a square adjacency matrix")

    # The starting state is always the supplied graph, normalized only into a
    # simple undirected graph.
    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)

    def common_counts(H):
        X = H.astype(np.int16, copy=False)
        return X @ X

    def is_valid(H):
        C = common_counts(H)
        return not np.any(np.triu(C > 1, 1))

    def repair(H):
        """
        Delete edges covering as many current common-neighbor violations as
        possible.  Random tie-breaking gives genuinely different repairs while
        remaining entirely input-derived.
        """
        H = H.copy()
        bound = n * (n - 1) // 2

        for _ in range(bound):
            C = common_counts(H)
            bad = C > 1
            np.fill_diagonal(bad, False)
            if not np.any(bad):
                return H

            # For an edge uv, bad@H counts bad pairs for which v is a
            # witness adjacent to u.  Both directions contribute to deletion
            # usefulness.
            support = bad.astype(np.int16) @ H.astype(np.int16)
            score = (support + support.T) * H
            best_score = int(score.max())

            if best_score > 0:
                choices = np.argwhere(np.triu(score == best_score, 1))
                if len(choices) > 1:
                    deg = H.sum(axis=1)
                    endpoint_load = deg[choices[:, 0]] + deg[choices[:, 1]]
                    choices = choices[endpoint_load == endpoint_load.max()]
                u, v = choices[int(rng.integers(len(choices)))]
            else:
                # Guaranteed conservative progress if numerical support ever
                # fails to identify an edge.
                pairs = np.argwhere(np.triu(bad, 1))
                i, j = pairs[int(rng.integers(len(pairs)))]
                shared = np.flatnonzero(H[i] & H[j])
                w = int(shared[int(rng.integers(len(shared)))])
                if H[i].sum() >= H[j].sum():
                    u, v = int(i), w
                else:
                    u, v = int(j), w

            H[u, v] = 0
            H[v, u] = 0

        # Defensive final repair, which also explicitly establishes validity.
        for _ in range(bound):
            C = common_counts(H)
            bad = np.argwhere(np.triu(C > 1, 1))
            if len(bad) == 0:
                break
            i, j = map(int, bad[0])
            shared = np.flatnonzero(H[i] & H[j])
            w = int(shared[0])
            H[i, w] = 0
            H[w, i] = 0

        return H

    def masks_from_matrix(H):
        masks = [0] * n
        for i in range(n):
            row = np.flatnonzero(H[i])
            m = 0
            for j in row:
                m |= 1 << int(j)
            masks[i] = m
        return masks

    def matrix_from_masks(masks):
        H = np.zeros((n, n), dtype=np.uint8)
        for i, m in enumerate(masks):
            x = m
            while x:
                bit = x & -x
                j = bit.bit_length() - 1
                H[i, j] = 1
                x ^= bit
        np.fill_diagonal(H, 0)
        return H

    def legal_add(masks, u, v):
        """Bitset test for whether an edge uv would create a 4-cycle."""
        nv = masks[v]
        x = masks[u]
        while x:
            bit = x & -x
            w = bit.bit_length() - 1
            if masks[w] & nv:
                return False
            x ^= bit
        return True

    def saturate(masks, degrees):
        """
        Repeatedly add a legal edge.  The choice is recomputed after every
        addition rather than using one static ordering, which more strongly
        maintains a balanced degree sequence.
        """
        max_additions = n * (n - 1) // 2
        for _ in range(max_additions):
            best_key = None
            tied = []

            for u in range(n - 1):
                mu = masks[u]
                for v in range(u + 1, n):
                    if (mu >> v) & 1:
                        continue
                    if not legal_add(masks, u, v):
                        continue

                    # Low degree sum is primary; low imbalance prevents a few
                    # vertices from becoming bottlenecks.  A tiny randomized
                    # tie choice supplies diverse exchange trajectories.
                    key = (
                        degrees[u] + degrees[v],
                        abs(degrees[u] - degrees[v]),
                        max(degrees[u], degrees[v]),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        tied = [(u, v)]
                    elif key == best_key:
                        tied.append((u, v))

            if not tied:
                break

            u, v = tied[int(rng.integers(len(tied)))]
            masks[u] |= 1 << v
            masks[v] |= 1 << u
            degrees[u] += 1
            degrees[v] += 1

        return masks, degrees

    def make_maximal(H):
        masks = masks_from_matrix(H)
        degrees = [m.bit_count() for m in masks]
        masks, degrees = saturate(masks, degrees)
        return masks, degrees

    # Two different repair trajectories are cheap at n=49 and substantially
    # reduce sensitivity to unlucky deletion ties in a dense input graph.
    best_masks = None
    best_degrees = None
    best_edges = -1

    for _ in range(2):
        repaired = repair(source)
        masks, degrees = make_maximal(repaired)
        candidate = matrix_from_masks(masks)

        # Explicit mandatory phase verification.
        if not is_valid(candidate):
            repaired = repair(candidate)
            masks, degrees = make_maximal(repaired)
            candidate = matrix_from_masks(masks)

        edge_count = int(candidate.sum() // 2)
        if edge_count > best_edges:
            best_masks = masks
            best_degrees = degrees
            best_edges = edge_count

    # Current can move across equal, or occasionally one-edge-worse, maximal
    # states.  The global best is retained, allowing shallow exchange valleys
    # without ever risking the final score.
    current_masks = best_masks[:]
    current_degrees = best_degrees[:]
    current_edges = best_edges

    for trial in range(18):
        masks = current_masks[:]
        degrees = current_degrees[:]

        edge_list = []
        for u in range(n - 1):
            x = masks[u] >> (u + 1)
            v = u + 1
            while x:
                if x & 1:
                    # Edges joining large neighborhoods tend to block many
                    # possible additions and are useful destruction targets.
                    value = (degrees[u] - 1) * (degrees[v] - 1)
                    edge_list.append((value, u, v))
                x >>= 1
                v += 1

        if not edge_list:
            break

        # Alternate modest and larger ruin sizes.  Avoid repeatedly selecting
        # exactly the same high-degree edge by sampling from a high-value pool.
        remove_count = 3 + (trial % 3)
        for r in range(remove_count):
            available = []
            for value, u, v in edge_list:
                if (masks[u] >> v) & 1:
                    available.append((value, u, v))
            if not available:
                break

            available.sort(reverse=True)
            pool_size = max(1, min(len(available), len(available) // 3 + 2))
            # Every fourth move broadens the pool for structural diversity.
            if (trial + r) % 4 == 0:
                pool_size = max(pool_size, min(len(available), len(available) // 2))
            _, u, v = available[int(rng.integers(pool_size))]

            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            degrees[u] -= 1
            degrees[v] -= 1

        masks, degrees = saturate(masks, degrees)
        H = matrix_from_masks(masks)

        # Explicit verification after every exchange/reconstruction phase.
        if not is_valid(H):
            H = repair(H)
            masks, degrees = make_maximal(H)
            H = matrix_from_masks(masks)

        edge_count = int(H.sum() // 2)
        if edge_count > best_edges:
            best_masks = masks[:]
            best_degrees = degrees[:]
            best_edges = edge_count

        # Equal moves permit exploration; a one-edge decline is allowed only
        # as a bounded temporary escape mechanism.
        if edge_count >= current_edges - 1:
            current_masks = masks[:]
            current_degrees = degrees[:]
            current_edges = edge_count
        elif best_edges > current_edges:
            current_masks = best_masks[:]
            current_degrees = best_degrees[:]
            current_edges = best_edges

    result = matrix_from_masks(best_masks)

    # Final mandatory conservative verification.
    if not is_valid(result):
        result = repair(result)
        masks, _ = make_maximal(result)
        result = matrix_from_masks(masks)

    return result.astype(np.uint8, copy=False)

# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
