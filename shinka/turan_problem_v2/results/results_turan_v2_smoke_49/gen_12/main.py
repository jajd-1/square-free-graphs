# EVOLVE-BLOCK-START
"""Bitset-based source-aware exchange search for dense C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Improve the supplied graph by retaining source edges, repairing all C4
    conflicts implicitly, and applying safe greedy augmentation plus bounded
    ruin-and-refill exchanges.

    The maintained invariant is that every vertex pair has at most one common
    neighbor.  The bitset insertion test is exactly the no-length-three-path
    condition for adding an edge to a C4-free graph.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    full = (1 << n) - 1
    iu, ju = np.triu_indices(n, 1)
    pair_count = len(iu)

    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)
    source_degree = source.sum(axis=1).astype(np.int16)
    source_upper = source[iu, ju].astype(bool)
    source_edges = np.flatnonzero(source_upper)

    def edge_total(masks):
        return sum(x.bit_count() for x in masks) // 2

    def can_add(masks, degrees, u, v):
        """True iff adding uv cannot complete a four-cycle."""
        # Iterating through the lower degree neighborhood is substantially
        # cheaper in the dense part of the search.
        if degrees[u] > degrees[v]:
            u, v = v, u
        neigh_v = masks[v]
        x = masks[u]
        while x:
            bit = x & -x
            w = bit.bit_length() - 1
            if masks[w] & neigh_v:
                return False
            x ^= bit
        return True

    def matrix_from_masks(masks):
        H = np.zeros((n, n), dtype=np.uint8)
        for u, bits in enumerate(masks):
            x = bits
            while x:
                b = x & -x
                v = b.bit_length() - 1
                H[u, v] = 1
                x ^= b
        np.fill_diagonal(H, 0)
        return H

    def masks_valid(masks):
        H = matrix_from_masks(masks).astype(np.int16)
        common = H @ H
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def saturate(masks, degrees, banned=None, source_bonus=True):
        """
        Safely add edges until maximality.  Low endpoint degree is the main
        objective; a small source bonus preserves input information without
        allowing input hubs to dominate the construction.
        """
        masks = masks[:]
        degrees = degrees[:]

        if banned is None:
            banned_set = set()
        else:
            banned_set = set(banned)

        # At most this many successful additions are possible.
        for _ in range(pair_count):
            best_score = None
            near_best = []

            for u in range(n - 1):
                absent = (full ^ masks[u]) & ~((1 << (u + 1)) - 1)
                absent &= ~(1 << u)

                while absent:
                    bit = absent & -absent
                    v = bit.bit_length() - 1
                    absent ^= bit

                    key = u * n + v
                    if key in banned_set:
                        continue
                    if not can_add(masks, degrees, u, v):
                        continue

                    du = degrees[u]
                    dv = degrees[v]

                    # Dense C4-free graphs tend to have degrees near 6--7.
                    # This score uses coarse buckets, intentionally leaving
                    # many ties for randomized diversification.
                    excess = max(du - 7, 0) ** 2 + max(dv - 7, 0) ** 2
                    score = 16 * (du + dv) + 7 * excess + du * dv

                    if source_bonus and source[u, v]:
                        score -= 2

                    if best_score is None or score < best_score:
                        best_score = score
                        near_best = [(u, v)]
                    elif score <= best_score + 2:
                        near_best.append((u, v))

            # Once every non-banned option is exhausted, permit removed edges
            # again.  This preserves maximality even if an exchange fails.
            if not near_best and banned_set:
                banned_set.clear()
                continue

            if not near_best:
                break

            u, v = near_best[int(rng.integers(len(near_best)))]
            # Explicit local assertion of the invariant before modification.
            if not can_add(masks, degrees, u, v):
                continue

            masks[u] |= 1 << v
            masks[v] |= 1 << u
            degrees[u] += 1
            degrees[v] += 1

        return masks, degrees

    def source_start(style):
        """
        Greedily retain a C4-free subgraph of the actual input.  Every style
        has a different source-edge ordering, so subsequent augmentation
        begins from materially input-dependent states.
        """
        masks = [0] * n
        degrees = [0] * n

        if source_edges.size == 0:
            return masks, degrees

        su = iu[source_edges]
        sv = ju[source_edges]
        load = source_degree[su].astype(np.int32) + source_degree[sv].astype(np.int32)
        product = source_degree[su].astype(np.int32) * source_degree[sv].astype(np.int32)
        noise = rng.random(source_edges.size)

        if style == 0:
            order = source_edges[np.lexsort((noise, load))]
        elif style == 1:
            order = source_edges[np.lexsort((noise, product))]
        elif style == 2:
            order = source_edges[np.lexsort((noise, -load))]
        elif style == 3:
            order = source_edges[np.lexsort((noise, -product))]
        else:
            order = source_edges[rng.permutation(source_edges.size)]

        for edge in order:
            u = int(iu[edge])
            v = int(ju[edge])
            if can_add(masks, degrees, u, v):
                masks[u] |= 1 << v
                masks[v] |= 1 << u
                degrees[u] += 1
                degrees[v] += 1

        return masks, degrees

    archive = []

    def archive_insert(masks, degrees):
        nonlocal archive
        if not masks_valid(masks):
            return

        signature = tuple(masks)
        for _, _, _, old_signature in archive:
            if signature == old_signature:
                return

        archive.append((edge_total(masks), masks[:], degrees[:], signature))
        archive.sort(key=lambda z: z[0], reverse=True)
        archive = archive[:5]

    # Five source-dependent repairs are retained from the prior configuration,
    # now with inexpensive bitset augmentation.
    for style in range(5):
        masks, degrees = source_start(style)
        masks, degrees = saturate(masks, degrees)
        archive_insert(masks, degrees)

    if not archive:
        masks = [0] * n
        degrees = [0] * n
        masks, degrees = saturate(masks, degrees, source_bonus=False)
        archive_insert(masks, degrees)

    # More exchange opportunities than the previous 14-move schedule.  The
    # sizes include shallow moves for local swaps and occasional deeper moves
    # for crossing saturation barriers.
    ruin_sizes = (
        2, 2, 3, 2, 3,
        4, 3, 4, 2, 5,
        3, 4, 5, 2, 4,
        6, 3, 5, 4, 3,
    )

    for trial, remove_count in enumerate(ruin_sizes):
        if len(archive) == 1 or rng.random() < 0.68:
            _, base_masks, base_degrees, _ = archive[0]
        else:
            rank = int(rng.integers(min(len(archive), 4)))
            _, base_masks, base_degrees, _ = archive[rank]

        masks = base_masks[:]
        degrees = base_degrees[:]

        present = []
        values = []
        for u in range(n - 1):
            x = masks[u] & ~((1 << (u + 1)) - 1)
            while x:
                bit = x & -x
                v = bit.bit_length() - 1
                x ^= bit
                present.append((u, v))
                # Edges between highly loaded endpoints normally constrain
                # many cross-neighborhood possibilities.
                values.append(
                    5 * (degrees[u] - 1) * (degrees[v] - 1)
                    + degrees[u] + degrees[v]
                )

        if len(present) < remove_count:
            continue

        values = np.asarray(values)
        # Broad pools retain stochasticity and avoid repeatedly deleting only
        # a single structurally central edge.
        percentile = 55 + 5 * (trial % 5)
        cutoff = np.percentile(values, percentile)
        pool = [k for k, value in enumerate(values) if value >= cutoff]
        if len(pool) < remove_count:
            pool = list(range(len(present)))

        selected = []
        used = set()

        for pick in range(remove_count):
            available = [k for k in pool if k not in selected]
            if not available:
                available = [k for k in range(len(present)) if k not in selected]

            # Most moves spread deletion over different endpoint regions.
            if pick and trial % 4 != 0:
                disjoint = [
                    k for k in available
                    if present[k][0] not in used and present[k][1] not in used
                ]
                if disjoint:
                    available = disjoint

            chosen = available[int(rng.integers(len(available)))]
            selected.append(chosen)
            used.add(present[chosen][0])
            used.add(present[chosen][1])

        banned = set()
        for idx in selected:
            u, v = present[idx]
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            degrees[u] -= 1
            degrees[v] -= 1
            banned.add(u * n + v)

        # Deletion cannot introduce a C4, but explicitly verify before the
        # refill phase as a conservative invariant guard.
        if not masks_valid(masks):
            continue

        masks, degrees = saturate(masks, degrees, banned=banned)
        archive_insert(masks, degrees)

    best_masks = archive[0][1][:]
    best_degrees = archive[0][2][:]

    # Mandatory final invariant verification.  This repair branch should
    # never run in normal operation because all additions use can_add.
    if not masks_valid(best_masks):
        for _ in range(pair_count):
            H = matrix_from_masks(best_masks).astype(np.int16)
            common = H @ H
            np.fill_diagonal(common, 0)
            bad = np.argwhere(np.triu(common > 1, 1))
            if bad.size == 0:
                break

            u, v = map(int, bad[0])
            shared_bits = best_masks[u] & best_masks[v]
            if not shared_bits:
                continue

            candidates = []
            x = shared_bits
            while x:
                bit = x & -x
                w = bit.bit_length() - 1
                x ^= bit
                candidates.append(w)

            w = max(candidates, key=lambda z: best_degrees[z])
            if best_degrees[u] >= best_degrees[v]:
                a, b = u, w
            else:
                a, b = v, w

            if (best_masks[a] >> b) & 1:
                best_masks[a] &= ~(1 << b)
                best_masks[b] &= ~(1 << a)
                best_degrees[a] -= 1
                best_degrees[b] -= 1

        if masks_valid(best_masks):
            best_masks, best_degrees = saturate(best_masks, best_degrees)

    result = matrix_from_masks(best_masks)

    # Strict final check; a zero graph is only a last-resort safety fallback.
    Hcheck = result.astype(np.int16)
    common = Hcheck @ Hcheck
    np.fill_diagonal(common, 0)
    if np.any(common > 1):
        result = np.zeros((n, n), dtype=np.uint8)

    return result.astype(np.uint8, copy=False)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
