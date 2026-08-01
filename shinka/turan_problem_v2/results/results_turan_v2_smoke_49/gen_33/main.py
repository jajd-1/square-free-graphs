# EVOLVE-BLOCK-START
"""Adaptive input-derived path-swap search for dense C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Improve an input graph by editing it into a dense C4-free graph.

    Every retained graph satisfies the invariant that each distinct pair of
    vertices has at most one common neighbor.  Initial states retain actual
    input edges; subsequent search consists only of legal additions and
    explicitly checked deletion/refill moves.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = iu.size

    # Treat a one-sided input edge as an undirected suggestion, but never keep
    # loops.  This is the only source preference used by the search.
    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)
    source_up = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int32)
    source_edges = np.flatnonzero(source_up)

    edge_id = -np.ones((n, n), dtype=np.int32)
    edge_id[iu, ju] = np.arange(m, dtype=np.int32)
    edge_id[ju, iu] = np.arange(m, dtype=np.int32)

    ARCHIVE_LIMIT = 9

    def count_edges(B):
        return int(B.sum() // 2)

    def valid(B):
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def legal_add(B, u, v):
        """Test the exact C4-free insertion condition for uv."""
        nu = np.flatnonzero(B[u])
        nv = np.flatnonzero(B[v])
        if nu.size == 0 or nv.size == 0:
            return True
        return not np.any(B[np.ix_(nu, nv)])

    def legal_information(B):
        """
        In a valid state, xy is insertable iff there is no length-three walk
        from x to y.  p3 also supplies obstruction information for swaps.
        """
        X = B.astype(np.int16)
        p3 = (X @ X) @ X
        legal = (B == 0) & (p3 == 0)
        np.fill_diagonal(legal, False)
        return legal, p3

    def saturate(B, banned=None, source_bias=0.35):
        """
        Greedily make a valid graph maximal.

        The score combines:
          - legal candidates destroyed through N(u) x N(v),
          - endpoint degree load,
          - a stronger penalty above the useful degree-seven plateau,
          - a small preference for source edges.

        Temporarily banning deleted edges makes ruin/refill moves seek genuine
        replacements before allowing a rollback.
        """
        B = B.copy()
        forbidden = (
            np.zeros(m, dtype=bool) if banned is None else banned.copy()
        )

        for _ in range(m):
            legal, _ = legal_information(B)
            available = np.flatnonzero(legal[iu, ju] & ~forbidden)

            if available.size == 0 and np.any(forbidden):
                forbidden[:] = False
                available = np.flatnonzero(legal[iu, ju])

            if available.size == 0:
                break

            X = B.astype(np.int16)
            L = legal.astype(np.int16)

            # Existing two-edge neighborhoods around a candidate estimate the
            # currently legal opportunities killed by selecting that edge.
            future_loss = (X @ L) @ X
            degree = B.sum(axis=1).astype(np.int32)

            au = iu[available]
            av = ju[available]
            loss = future_loss[au, av].astype(np.int32)
            load = degree[au] + degree[av]
            excess = (
                np.maximum(degree[au] - 7, 0)
                + np.maximum(degree[av] - 7, 0)
            )
            imbalance = np.abs(degree[au] - degree[av])

            score = 5 * loss + load + 2 * excess + imbalance // 2
            low = score.min()

            # Randomness is restricted to a narrow quality band so different
            # input matrices and seeds still explore distinct basins.
            band = available[score <= low + 1]
            if band.size == 0:
                band = available[score == low]

            preferred = band[source_up[band]]
            if preferred.size and rng.random() < source_bias:
                band = preferred

            chosen = int(band[int(rng.integers(band.size))])
            u = int(iu[chosen])
            v = int(ju[chosen])

            # Redundant explicit test protects the invariant even if scoring
            # code is changed later.
            if legal_add(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def source_order(style):
        """Several input-sensitive edge orderings for C4-free repairs."""
        if source_edges.size == 0:
            return source_edges

        u = iu[source_edges]
        v = ju[source_edges]
        du = source_degree[u]
        dv = source_degree[v]
        load = du + dv
        product = du * dv
        gap = np.abs(du - dv)

        # Neighbor overlap distinguishes source edges inside dense regions
        # from edges likely to be useful in a broad sparse scaffold.
        overlap = np.sum(source[u] & source[v], axis=1)
        noise = rng.random(source_edges.size)

        if style == 0:
            return source_edges[np.lexsort((noise, load))]
        if style == 1:
            return source_edges[np.lexsort((noise, product))]
        if style == 2:
            return source_edges[np.lexsort((noise, overlap))]
        if style == 3:
            return source_edges[np.lexsort((noise, gap))]
        if style == 4:
            return source_edges[np.lexsort((noise, -load))]
        if style == 5:
            return source_edges[np.lexsort((noise, -overlap))]
        if style == 6:
            return source_edges[np.lexsort((noise, -gap))]

        return source_edges[rng.permutation(source_edges.size)]

    def scaffold(style, limit):
        """Build a C4-free subset consisting exclusively of input edges."""
        B = np.zeros((n, n), dtype=np.uint8)
        order = source_order(style)
        if limit is not None:
            order = order[:min(int(limit), order.size)]

        for e in order:
            u = int(iu[e])
            v = int(ju[e])
            if legal_add(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def add_archive(archive, B):
        """Keep only distinct valid elite states."""
        if not valid(B):
            return archive

        sig = B[iu, ju].tobytes()
        for _, _, old_sig in archive:
            if sig == old_sig:
                return archive

        archive.append((count_edges(B), B.copy(), sig))
        archive.sort(key=lambda x: x[0], reverse=True)
        return archive[:ARCHIVE_LIMIT]

    def unlock_scores(B, present):
        """
        Score existing edges by the number of uniquely blocked absent pairs
        they can unlock.

        If xy has exactly one x-a-b-y length-three path, deleting xa, ab, or
        by makes xy immediately legal.  This is used for directed ruin moves
        rather than choosing deletions solely by endpoint degree.
        """
        _, p3 = legal_information(B)
        credits = np.zeros(m, dtype=np.int32)
        blocked = np.argwhere(np.triu((B == 0) & (p3 == 1), 1))

        for x, y in blocked:
            x = int(x)
            y = int(y)
            nx = np.flatnonzero(B[x])
            ny = np.flatnonzero(B[y])

            if nx.size == 0 or ny.size == 0:
                continue

            aa = nx[np.any(B[np.ix_(nx, ny)], axis=1)]
            if aa.size != 1:
                continue

            a = int(aa[0])
            bb = ny[B[a, ny] != 0]
            if bb.size != 1:
                continue

            b = int(bb[0])

            e1 = int(edge_id[x, a])
            e2 = int(edge_id[a, b])
            e3 = int(edge_id[b, y])
            if e1 >= 0:
                credits[e1] += 1
            if e2 >= 0:
                credits[e2] += 1
            if e3 >= 0:
                credits[e3] += 1

        deg = B.sum(axis=1).astype(np.int32)
        endpoint_load = deg[iu[present]] + deg[ju[present]]
        endpoint_excess = (
            np.maximum(deg[iu[present]] - 7, 0)
            + np.maximum(deg[ju[present]] - 7, 0)
        )

        # Large unlock credit identifies real replacement opportunities;
        # degree terms make hub dismantling useful even when credits tie.
        return 20 * credits[present] + endpoint_load + 3 * endpoint_excess

    archive = []

    # Full source repairs preserve broad input structure.  Partial repairs
    # retain genuine source information but leave room for balanced growth.
    plans = (
        (0, None), (1, None), (2, None), (3, None),
        (4, None), (5, None),
        (0, 24), (2, 34), (3, 44), (6, 54),
    )

    for style, limit in plans:
        B = scaffold(style, limit)
        if valid(B):
            B = saturate(B)
            if valid(B):
                archive = add_archive(archive, B)

    # Empty input has no source scaffold, so begin from the edit state with no
    # retained input edges only as an unavoidable fallback.
    if not archive:
        B = np.zeros((n, n), dtype=np.uint8)
        B = saturate(B, source_bias=0.0)
        archive = add_archive(archive, B)

    # Adaptive path-unlocking swaps.  Small removals preserve most of a good
    # state; periodic larger moves escape maximal but inferior configurations.
    schedule = (
        (1, 88), (1, 78), (2, 82), (2, 72),
        (2, 62), (3, 76), (3, 64), (3, 52),
        (4, 70), (4, 56), (2, 48), (3, 42),
        (4, 44), (5, 52), (3, 34), (2, 58),
        (4, 36), (5, 42),
    )

    for trial, (remove_count, percentile) in enumerate(schedule):
        if len(archive) == 1 or rng.random() < 0.60:
            base = archive[0][1]
        else:
            rank_limit = min(5, len(archive))
            base = archive[int(rng.integers(rank_limit))][1]

        present = np.flatnonzero(base[iu, ju])
        if present.size < remove_count:
            continue

        scores = unlock_scores(base, present)
        threshold = np.percentile(scores, percentile)
        pool = present[scores >= threshold]
        if pool.size < remove_count:
            pool = present

        selected = []
        used_vertices = set()

        for pick in range(remove_count):
            remaining = pool[~np.isin(pool, selected)]
            if remaining.size == 0:
                remaining = present[~np.isin(present, selected)]

            # Except for dedicated concentrated moves, spread deletions so
            # each one opens a different local obstruction region.
            if pick > 0 and trial % 5 != 0:
                spread = np.array(
                    [
                        e for e in remaining
                        if int(iu[e]) not in used_vertices
                        and int(ju[e]) not in used_vertices
                    ],
                    dtype=np.int64,
                )
                if spread.size:
                    remaining = spread

            e = int(remaining[int(rng.integers(remaining.size))])
            selected.append(e)
            used_vertices.add(int(iu[e]))
            used_vertices.add(int(ju[e]))

        candidate = base.copy()
        banned = np.zeros(m, dtype=bool)

        for e in selected:
            u = int(iu[e])
            v = int(ju[e])
            candidate[u, v] = 0
            candidate[v, u] = 0
            banned[e] = True

        if not valid(candidate):
            continue

        candidate = saturate(candidate, banned=banned, source_bias=0.25)

        if valid(candidate):
            archive = add_archive(archive, candidate)

    best = archive[0][1].copy()

    # Conservative deletion-only repair is retained as a final guard.
    for _ in range(n * n):
        X = best.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        bad = np.argwhere(np.triu(common > 1, 1))
        if bad.size == 0:
            break

        u, v = map(int, bad[0])
        shared = np.flatnonzero(best[u] & best[v])
        if shared.size == 0:
            continue

        deg = best.sum(axis=1)
        x = int(shared[np.argmax(deg[shared])])
        if deg[u] >= deg[v]:
            best[u, x] = 0
            best[x, u] = 0
        else:
            best[v, x] = 0
            best[x, v] = 0

    # A final saturation is safe only after explicit validity confirmation.
    if valid(best):
        best = saturate(best, source_bias=0.20)

    # Strict final fallback.  This can only delete edges and therefore always
    # terminates with a valid graph.
    if not valid(best):
        for _ in range(n * n):
            X = best.astype(np.int16)
            common = X @ X
            np.fill_diagonal(common, 0)
            bad = np.argwhere(np.triu(common > 1, 1))
            if bad.size == 0:
                break

            u, v = map(int, bad[0])
            shared = np.flatnonzero(best[u] & best[v])
            if shared.size:
                x = int(shared[0])
                best[u, x] = 0
                best[x, u] = 0

    np.fill_diagonal(best, 0)
    return best.astype(np.uint8)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
