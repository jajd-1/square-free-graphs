# EVOLVE-BLOCK-START
"""Portfolio ruin-and-refill search for dense C4-free graph editing."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Convert an arbitrary uint8 adjacency matrix into a dense C4-free graph.

    The search is deliberately input-aware:
      * every nonempty initial scaffold is a retained subset of input edges;
      * input edges are mildly preferred during equivalent safe insertions;
      * several differently ordered input repairs form a diverse state beam;
      * elite states are perturbed by obstruction-aware edge deletions and
        safely refilled.

    All accepted states satisfy the invariant that no two vertices have more
    than one common neighbor.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = iu.size

    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)

    source_upper = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int32)
    source_edges = np.flatnonzero(source_upper)

    edge_id = -np.ones((n, n), dtype=np.int32)
    edge_id[iu, ju] = np.arange(m, dtype=np.int32)
    edge_id[ju, iu] = np.arange(m, dtype=np.int32)

    BEAM_LIMIT = 12

    def edge_count(B):
        return int(B.sum() // 2)

    def is_valid(B):
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def insertion_legal(B, u, v):
        """Test whether uv closes a four-cycle in the current valid graph."""
        nu = np.flatnonzero(B[u])
        nv = np.flatnonzero(B[v])
        if nu.size == 0 or nv.size == 0:
            return True
        return not np.any(B[np.ix_(nu, nv)])

    def legality(B):
        """
        For a C4-free graph, an absent pair is insertable iff it has no
        length-three walk.  p3 also describes deletion obstructions.
        """
        X = B.astype(np.int16)
        p3 = (X @ X) @ X
        legal = (B == 0) & (p3 == 0)
        np.fill_diagonal(legal, False)
        return legal, p3

    def complete(B, mode=0, banned=None):
        """
        Safely greedily saturate B.

        Candidate score combines exact presently-legal future loss with a
        degree balancing pressure.  Deleted edges can be temporarily banned,
        forcing refill to seek alternate structural opportunities first.
        """
        B = B.copy()
        forbidden = (
            np.zeros(m, dtype=bool) if banned is None else banned.copy()
        )

        for _ in range(m):
            legal, _ = legality(B)
            available = np.flatnonzero(legal[iu, ju] & ~forbidden)

            if available.size == 0 and np.any(forbidden):
                forbidden[:] = False
                available = np.flatnonzero(legal[iu, ju])

            if available.size == 0:
                break

            X = B.astype(np.int16)
            L = legal.astype(np.int16)
            loss_matrix = (X @ L) @ X
            degree = B.sum(axis=1).astype(np.int32)

            au = iu[available]
            av = ju[available]
            loss = loss_matrix[au, av].astype(np.int32)
            load = degree[au] + degree[av]
            high = (
                np.maximum(degree[au] - 7, 0)
                + np.maximum(degree[av] - 7, 0)
            )
            imbalance = np.abs(degree[au] - degree[av])

            if mode == 0:
                score = 5 * loss + load + high
                width = 1
            elif mode == 1:
                score = 4 * loss + 2 * load + 2 * high
                width = 2
            else:
                score = 4 * loss + load + imbalance + 3 * high
                width = 1

            best_score = score.min()
            band = available[score <= best_score + width]
            if band.size == 0:
                band = available[score == best_score]

            # Retaining an original edge breaks ties without overwhelming the
            # structural low-loss choice.
            original = band[source_upper[band]]
            if original.size and rng.random() < 0.42:
                band = original

            chosen = int(band[int(rng.integers(band.size))])
            u = int(iu[chosen])
            v = int(ju[chosen])

            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def source_order(style):
        """Generate substantially different input-dependent source orderings."""
        if source_edges.size == 0:
            return source_edges

        u = iu[source_edges]
        v = ju[source_edges]
        du = source_degree[u]
        dv = source_degree[v]
        load = du + dv
        prod = du * dv
        diff = np.abs(du - dv)
        noise = rng.random(source_edges.size)

        if style == 0:
            key = load
        elif style == 1:
            key = -load
        elif style == 2:
            key = prod
        elif style == 3:
            key = -prod
        elif style == 4:
            key = diff
        elif style == 5:
            key = -diff
        elif style == 6:
            key = 2 * load + diff
        elif style == 7:
            key = -(2 * load - diff)
        else:
            return source_edges[rng.permutation(source_edges.size)]

        return source_edges[np.lexsort((noise, key))]

    def scaffold(style, prefix):
        """Build a valid graph using only actual source edges."""
        B = np.zeros((n, n), dtype=np.uint8)
        order = source_order(style)

        if prefix is not None:
            order = order[:min(int(prefix), order.size)]

        for e in order:
            u = int(iu[e])
            v = int(ju[e])
            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def add_to_beam(beam, B):
        """Insert one distinct verified solution into the elite portfolio."""
        if not is_valid(B):
            return beam

        signature = B[iu, ju].tobytes()
        for _, _, old_signature in beam:
            if old_signature == signature:
                return beam

        beam.append((edge_count(B), B.copy(), signature))
        beam.sort(key=lambda item: item[0], reverse=True)
        return beam[:BEAM_LIMIT]

    def deletion_scores(B, present):
        """
        Score deletions by exact unique length-three-path unlock credit.

        An absent xy with exactly one length-three path x-a-b-y becomes legal
        after deleting any edge of that path.  Edges with many such credits
        are especially useful ruin candidates.  Degree pressure breaks ties
        toward reducing overloaded regions.
        """
        _, p3 = legality(B)
        credit = np.zeros(m, dtype=np.int32)
        blocked = np.argwhere(np.triu((B == 0) & (p3 == 1), 1))

        for x, y in blocked:
            x = int(x)
            y = int(y)
            nx = np.flatnonzero(B[x])
            ny = np.flatnonzero(B[y])

            if nx.size == 0 or ny.size == 0:
                continue

            left = nx[np.any(B[np.ix_(nx, ny)], axis=1)]
            if left.size != 1:
                continue

            a = int(left[0])
            right = ny[B[a, ny] != 0]
            if right.size != 1:
                continue

            b = int(right[0])

            e1 = int(edge_id[x, a])
            e2 = int(edge_id[a, b])
            e3 = int(edge_id[b, y])

            if e1 >= 0:
                credit[e1] += 1
            if e2 >= 0:
                credit[e2] += 1
            if e3 >= 0:
                credit[e3] += 1

        degree = B.sum(axis=1).astype(np.int32)
        endpoint_load = degree[iu[present]] + degree[ju[present]]
        return 18 * credit[present] + endpoint_load

    beam = []

    # Input-derived portfolio.  Prefix states avoid allowing a dense source to
    # dictate all early decisions, while full scans preserve useful structure.
    plan = (
        (0, None, 0), (1, None, 1), (2, None, 2), (3, None, 0),
        (4, None, 1), (5, None, 2), (6, None, 0),
        (0, 18, 1), (1, 28, 2), (2, 38, 0),
        (4, 48, 1), (8, 58, 2),
    )

    for style, prefix, mode in plan:
        B = scaffold(style, prefix)
        if is_valid(B):
            B = complete(B, mode=mode)
            if is_valid(B):
                beam = add_to_beam(beam, B)

    # A source-free matrix still needs a valid graph; all normal nonempty
    # cases enter through source-derived scaffolds above.
    if not beam:
        B = complete(np.zeros((n, n), dtype=np.uint8), mode=0)
        beam = add_to_beam(beam, B)

    # Bounded ruin/refill trials.  Small ruins produce local exchanges and
    # larger ones occasionally escape a saturated but poorly balanced basin.
    schedule = (
        (1, 88, 0), (2, 82, 1), (2, 74, 2), (2, 66, 0),
        (3, 78, 1), (3, 70, 2), (3, 62, 0), (3, 54, 1),
        (4, 72, 2), (4, 64, 0), (4, 56, 1), (4, 48, 2),
        (5, 66, 0), (5, 56, 1), (5, 46, 2),
        (2, 58, 0), (3, 44, 1), (4, 40, 2),
        (5, 38, 0), (3, 32, 1), (4, 28, 2),
    )

    for trial, (remove_count, percentile, mode) in enumerate(schedule):
        if len(beam) == 1 or rng.random() < 0.58:
            base = beam[0][1]
        else:
            upper = min(6, len(beam))
            base = beam[int(rng.integers(upper))][1]

        current = base.copy()
        present = np.flatnonzero(current[iu, ju])
        if present.size < remove_count:
            continue

        scores = deletion_scores(current, present)
        cutoff = np.percentile(scores, percentile)
        pool = present[scores >= cutoff]
        if pool.size < remove_count:
            pool = present

        selected = []
        used = set()

        for pick in range(remove_count):
            remaining = pool[~np.isin(pool, selected)]
            if remaining.size == 0:
                remaining = present[~np.isin(present, selected)]

            # Most ruins spread over independent gadgets.  Some deliberately
            # concentrated trials dismantle a single difficult obstruction.
            if pick > 0 and trial % 5 != 0:
                independent = np.array(
                    [
                        e for e in remaining
                        if int(iu[e]) not in used and int(ju[e]) not in used
                    ],
                    dtype=np.int64,
                )
                if independent.size:
                    remaining = independent

            e = int(remaining[int(rng.integers(remaining.size))])
            selected.append(e)
            used.add(int(iu[e]))
            used.add(int(ju[e]))

        banned = np.zeros(m, dtype=bool)
        for e in selected:
            u = int(iu[e])
            v = int(ju[e])
            current[u, v] = 0
            current[v, u] = 0
            banned[e] = True

        if not is_valid(current):
            continue

        candidate = complete(current, mode=mode, banned=banned)
        if is_valid(candidate):
            beam = add_to_beam(beam, candidate)

    best = beam[0][1].copy()

    # Explicit final verification and conservative deletion-only repair.
    repair_budget = n * n
    for _ in range(repair_budget):
        X = best.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        bad = np.argwhere(np.triu(common > 1, 1))

        if bad.size == 0:
            break

        u, v = map(int, bad[0])
        common_neighbors = np.flatnonzero(best[u] & best[v])
        if common_neighbors.size == 0:
            continue

        degree = best.sum(axis=1)
        w = int(common_neighbors[np.argmax(degree[common_neighbors])])

        if degree[u] >= degree[v]:
            best[u, w] = 0
            best[w, u] = 0
        else:
            best[v, w] = 0
            best[w, v] = 0

    if is_valid(best):
        best = complete(best, mode=0)

    # Absolute safety fallback.
    if not is_valid(best):
        for _ in range(repair_budget):
            X = best.astype(np.int16)
            common = X @ X
            np.fill_diagonal(common, 0)
            bad = np.argwhere(np.triu(common > 1, 1))
            if bad.size == 0:
                break

            u, v = map(int, bad[0])
            common_neighbors = np.flatnonzero(best[u] & best[v])
            if common_neighbors.size:
                w = int(common_neighbors[0])
                best[u, w] = 0
                best[w, u] = 0

    np.fill_diagonal(best, 0)
    return best.astype(np.uint8)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
