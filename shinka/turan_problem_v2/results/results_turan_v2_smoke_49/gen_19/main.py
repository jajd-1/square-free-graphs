# EVOLVE-BLOCK-START
"""Input-aware repair, saturation, and bounded beam local search for C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Transform the supplied graph into a dense C4-free graph.

    The algorithm keeps the input graph as its starting state: it repairs
    common-neighbor violations by deleting highly offending input edges, then
    performs exact C4-safe greedy additions and small ruin-and-refill moves.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]

    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)

    def common_counts(H):
        X = H.astype(np.int16, copy=False)
        return X @ X

    def valid_matrix(H):
        C = common_counts(H)
        return not np.any(np.triu(C > 1, 1))

    def repair(H):
        """
        Delete edges until all pairs have at most one common neighbor.
        Edges adjacent to many currently violating pairs are preferred.
        """
        H = H.copy()
        limit = n * (n - 1) // 2

        for _ in range(limit):
            C = common_counts(H)
            bad = C > 1
            np.fill_diagonal(bad, False)

            if not np.any(bad):
                return H

            X = H.astype(np.int16, copy=False)
            support = bad.astype(np.int16) @ X
            score = (support + support.T) * X
            best = int(score.max())

            if best > 0:
                choices = np.argwhere(np.triu(score == best, 1))

                if len(choices) > 1:
                    deg = H.sum(axis=1)
                    loads = deg[choices[:, 0]] + deg[choices[:, 1]]
                    choices = choices[loads == loads.max()]

                u, v = choices[int(rng.integers(len(choices)))]
                u, v = int(u), int(v)
            else:
                pairs = np.argwhere(np.triu(bad, 1))
                i, j = pairs[int(rng.integers(len(pairs)))]
                common = np.flatnonzero(H[i] & H[j])

                if len(common) == 0:
                    continue

                w = int(common[int(rng.integers(len(common)))])
                if H[i].sum() >= H[j].sum():
                    u, v = int(i), w
                else:
                    u, v = int(j), w

            H[u, v] = 0
            H[v, u] = 0

        # Conservative deletion-only finishing pass.
        for _ in range(limit):
            C = common_counts(H)
            pairs = np.argwhere(np.triu(C > 1, 1))
            if len(pairs) == 0:
                break

            i, j = map(int, pairs[0])
            common = np.flatnonzero(H[i] & H[j])
            if len(common) == 0:
                continue

            w = int(common[0])
            if H[i].sum() >= H[j].sum():
                H[i, w] = 0
                H[w, i] = 0
            else:
                H[j, w] = 0
                H[w, j] = 0

        return H

    def to_masks(H):
        masks = [0] * n
        for u in range(n):
            mask = 0
            for v in np.flatnonzero(H[u]):
                mask |= 1 << int(v)
            masks[u] = mask
        return masks

    def from_masks(masks):
        H = np.zeros((n, n), dtype=np.uint8)
        for u, mask in enumerate(masks):
            x = mask
            while x:
                bit = x & -x
                v = bit.bit_length() - 1
                H[u, v] = 1
                x ^= bit
        np.fill_diagonal(H, 0)
        return H

    def legal_add(masks, u, v):
        """
        uv is legal iff no old edge joins N(u) to N(v).  Such an old edge
        would form a length-three path completed into a C4 by uv.
        """
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
        Repeatedly add a legal missing edge.  Low-degree endpoint preference
        encourages the nearly regular degree distribution of dense C4-free
        graphs while still using random tie breaks for search diversity.
        """
        limit = n * (n - 1) // 2

        for _ in range(limit):
            best_key = None
            tied = []

            for u in range(n - 1):
                mu = masks[u]
                du = degrees[u]

                for v in range(u + 1, n):
                    if (mu >> v) & 1:
                        continue
                    if not legal_add(masks, u, v):
                        continue

                    dv = degrees[v]
                    excess = max(du - 7, 0) ** 2 + max(dv - 7, 0) ** 2
                    key = (
                        du + dv + 2 * excess,
                        du * dv,
                        abs(du - dv),
                        max(du, dv),
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

    def maximize(H):
        masks = to_masks(H)
        degrees = [m.bit_count() for m in masks]
        masks, degrees = saturate(masks, degrees)
        return masks, degrees

    def edge_count(masks):
        return sum(m.bit_count() for m in masks) // 2

    def signature(masks):
        return tuple(masks)

    def edge_values(masks, degrees):
        items = []

        for u in range(n - 1):
            x = masks[u] >> (u + 1)
            v = u + 1

            while x:
                if x & 1:
                    # Higher values identify edges likely to constrain many
                    # cross-neighborhood possibilities.
                    value = (
                        (degrees[u] - 1) * (degrees[v] - 1)
                        + degrees[u]
                        + degrees[v]
                    )
                    items.append((value, u, v))
                x >>= 1
                v += 1

        return items

    def add_beam_item(beam, masks, degrees, limit=4):
        H = from_masks(masks)
        if not valid_matrix(H):
            return beam

        sig = signature(masks)
        for _, old_masks, _ in beam:
            if signature(old_masks) == sig:
                return beam

        beam.append((edge_count(masks), masks[:], degrees[:]))
        beam.sort(key=lambda item: item[0], reverse=True)
        return beam[:limit]

    beam = []
    best_masks = None
    best_edges = -1

    # Multiple repair trajectories retain actual input edges but use distinct
    # randomized tie decisions, producing varied dense C4-free starting states.
    for _ in range(5):
        repaired = repair(source)
        masks, degrees = maximize(repaired)
        H = from_masks(masks)

        if not valid_matrix(H):
            repaired = repair(H)
            masks, degrees = maximize(repaired)
            H = from_masks(masks)

        if valid_matrix(H):
            beam = add_beam_item(beam, masks, degrees)

            count = edge_count(masks)
            if count > best_edges:
                best_edges = count
                best_masks = masks[:]

    if not beam:
        masks, degrees = maximize(np.zeros((n, n), dtype=np.uint8))
        beam = add_beam_item(beam, masks, degrees)
        best_masks = masks[:]
        best_edges = edge_count(masks)

    # Bounded beam ruin-and-refill search.
    remove_schedule = (2, 3, 3, 4, 4, 5, 3, 5)

    for round_id, remove_count in enumerate(remove_schedule):
        proposals = []

        for rank, (_, base_masks, base_degrees) in enumerate(beam):
            attempts = 2 if rank < 2 else 1

            for attempt in range(attempts):
                masks = base_masks[:]
                degrees = base_degrees[:]
                edges = edge_values(masks, degrees)

                if len(edges) < remove_count:
                    continue

                edges.sort(reverse=True)

                if (round_id + attempt) % 3 == 0:
                    pool_size = max(remove_count, len(edges) // 2)
                else:
                    pool_size = max(remove_count, len(edges) // 4 + 3)

                pool = edges[:pool_size]
                chosen = []
                used = set()

                for r in range(remove_count):
                    available = [
                        item for item in pool
                        if (item[1], item[2]) not in chosen
                    ]

                    if not available:
                        available = [
                            item for item in edges
                            if (item[1], item[2]) not in chosen
                        ]

                    if r and (round_id + attempt) % 2 == 0:
                        disjoint = [
                            item for item in available
                            if item[1] not in used and item[2] not in used
                        ]
                        if disjoint:
                            available = disjoint

                    _, u, v = available[int(rng.integers(len(available)))]
                    chosen.append((u, v))
                    used.add(u)
                    used.add(v)

                for u, v in chosen:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    degrees[u] -= 1
                    degrees[v] -= 1

                H = from_masks(masks)
                if not valid_matrix(H):
                    continue

                masks, degrees = saturate(masks, degrees)
                H = from_masks(masks)

                if not valid_matrix(H):
                    H = repair(H)
                    masks, degrees = maximize(H)
                    H = from_masks(masks)

                if valid_matrix(H):
                    proposals.append((edge_count(masks), masks[:], degrees[:]))

                    count = edge_count(masks)
                    if count > best_edges:
                        best_edges = count
                        best_masks = masks[:]

        combined = beam + proposals
        combined.sort(key=lambda item: item[0], reverse=True)

        next_beam = []
        seen = set()

        for count, masks, degrees in combined:
            sig = signature(masks)
            if sig in seen:
                continue

            H = from_masks(masks)
            if not valid_matrix(H):
                continue

            seen.add(sig)
            next_beam.append((count, masks[:], degrees[:]))

            if len(next_beam) >= 4:
                break

        if next_beam:
            beam = next_beam

    result = from_masks(best_masks)

    # Mandatory final verification and conservative repair fallback.
    if not valid_matrix(result):
        result = repair(result)
        masks, _ = maximize(result)
        result = from_masks(masks)

    X = result.astype(np.int16, copy=False)
    C = X @ X
    np.fill_diagonal(C, 0)

    if np.any(C > 1):
        # Deletion-only emergency fallback.
        result = repair(result)

    return result.astype(np.uint8, copy=False)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
