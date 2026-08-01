# EVOLVE-BLOCK-START
"""Input-aware exact-unlock archive search for dense C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Edit the supplied matrix into a dense C4-free graph.

    Every retained state satisfies the invariant that every pair of distinct
    vertices has at most one common neighbor.  Construction begins from
    differently ordered subsets of the actual input edges, then safely adds
    edges and uses deletion/refill moves to escape maximal local states.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    pair_count = iu.size

    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)
    source_upper = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int32)
    source_edges = np.flatnonzero(source_upper)

    edge_index = -np.ones((n, n), dtype=np.int32)
    edge_index[iu, ju] = np.arange(pair_count, dtype=np.int32)
    edge_index[ju, iu] = np.arange(pair_count, dtype=np.int32)

    ARCHIVE_LIMIT = 10

    def edge_count(B):
        return int(B.sum() // 2)

    def valid(B):
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def insertion_legal(B, u, v):
        """Check whether adding uv preserves absence of four-cycles."""
        nu = np.flatnonzero(B[u])
        nv = np.flatnonzero(B[v])
        if nu.size == 0 or nv.size == 0:
            return True
        return not np.any(B[np.ix_(nu, nv)])

    def legal_data(B):
        """
        In a C4-free state, uv is insertable exactly when there is no
        length-three walk from u to v.
        """
        X = B.astype(np.int16)
        p3 = (X @ X) @ X
        legal = (B == 0) & (p3 == 0)
        np.fill_diagonal(legal, False)
        return legal, p3

    def saturate(B, banned=None):
        """
        Complete a valid state greedily.

        The future-loss score estimates how many currently legal candidate
        edges are made impossible by inserting a particular edge.  Lower loss
        is preferred; endpoint load is only a secondary balancing term.
        """
        B = B.copy()
        forbidden = (
            np.zeros(pair_count, dtype=bool)
            if banned is None else banned.copy()
        )

        for _ in range(pair_count):
            legal, _ = legal_data(B)
            available = np.flatnonzero(legal[iu, ju] & ~forbidden)

            if available.size == 0 and np.any(forbidden):
                forbidden[:] = False
                available = np.flatnonzero(legal[iu, ju])

            if available.size == 0:
                break

            X = B.astype(np.int16)
            L = legal.astype(np.int16)
            future_loss = (X @ L) @ X
            degree = B.sum(axis=1).astype(np.int32)

            au = iu[available]
            av = ju[available]
            loss = future_loss[au, av].astype(np.int32)
            load = degree[au] + degree[av]

            score = 4 * loss + load
            low = score.min()

            band = available[score <= low + 1]
            if band.size == 0:
                band = available[score == low]

            preferred = band[source_upper[band]]
            if preferred.size and rng.random() < 0.40:
                band = preferred

            chosen = int(band[int(rng.integers(band.size))])
            u = int(iu[chosen])
            v = int(ju[chosen])

            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def source_order(style):
        """Produce genuinely input-dependent source-edge orders."""
        if source_edges.size == 0:
            return source_edges

        du = source_degree[iu[source_edges]]
        dv = source_degree[ju[source_edges]]
        load = du + dv
        product = du * dv
        imbalance = np.abs(du - dv)
        noise = rng.random(source_edges.size)

        if style == 0:
            return source_edges[np.lexsort((noise, load))]
        if style == 1:
            return source_edges[np.lexsort((noise, -load))]
        if style == 2:
            return source_edges[np.lexsort((noise, product))]
        if style == 3:
            return source_edges[np.lexsort((noise, -product))]
        if style == 4:
            return source_edges[np.lexsort((noise, imbalance))]
        if style == 5:
            return source_edges[np.lexsort((noise, -imbalance))]
        return source_edges[rng.permutation(source_edges.size)]

    def build_scaffold(style, limit=None):
        """
        Retain a C4-free subset of actual source edges before augmentation.
        Bounded scaffolds preserve input dependence while exploring basins that
        do not become dominated by dense portions of the original matrix.
        """
        B = np.zeros((n, n), dtype=np.uint8)
        order = source_order(style)

        if limit is not None:
            order = order[:min(int(limit), order.size)]

        for e in order:
            u = int(iu[e])
            v = int(ju[e])
            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def archive_insert(archive, candidate):
        """Keep distinct valid elite states."""
        if not valid(candidate):
            return archive

        signature = candidate[iu, ju].tobytes()
        for _, _, old_signature in archive:
            if signature == old_signature:
                return archive

        archive.append((edge_count(candidate), candidate.copy(), signature))
        archive.sort(key=lambda item: item[0], reverse=True)
        return archive[:ARCHIVE_LIMIT]

    def obstruction_scores(B, present):
        """
        Exact unlock credit for present edges.

        For every absent pair x-y having one unique length-three path
        x-a-b-y, deletion of xa, ab, or by makes x-y legal.  Credit every
        edge on that unique path.  This directly measures useful deletions.
        """
        _, p3 = legal_data(B)
        credits = np.zeros(pair_count, dtype=np.int32)

        blocked = np.argwhere(np.triu((B == 0) & (p3 == 1), 1))

        for x, y in blocked:
            x = int(x)
            y = int(y)
            nx = np.flatnonzero(B[x])
            ny = np.flatnonzero(B[y])

            if nx.size == 0 or ny.size == 0:
                continue

            a_candidates = nx[np.any(B[np.ix_(nx, ny)], axis=1)]
            if a_candidates.size != 1:
                continue

            a = int(a_candidates[0])
            b_candidates = ny[B[a, ny] != 0]
            if b_candidates.size != 1:
                continue

            b = int(b_candidates[0])

            credits[int(edge_index[x, a])] += 1
            credits[int(edge_index[a, b])] += 1
            credits[int(edge_index[b, y])] += 1

        degree = B.sum(axis=1).astype(np.int32)
        score = 16 * credits[present]
        score += degree[iu[present]] + degree[ju[present]]
        return score

    archive = []

    # Full source repairs plus bounded source-derived scaffolds.
    scaffold_plan = (
        (0, None), (1, None), (2, None),
        (3, None), (4, None), (5, None),
        (0, 20), (1, 28), (2, 38), (6, 50),
    )

    for style, limit in scaffold_plan:
        state = build_scaffold(style, limit)
        if valid(state):
            state = saturate(state)
            if valid(state):
                archive = archive_insert(archive, state)

    # An empty input has no source-derived starting edge; use a safe fallback.
    if not archive:
        fallback = np.zeros((n, n), dtype=np.uint8)
        archive = archive_insert(archive, saturate(fallback))

    kick_schedule = (
        (2, 80), (2, 74), (2, 68), (2, 62),
        (3, 74), (3, 68), (3, 62), (3, 56),
        (3, 50), (3, 44),
        (4, 66), (4, 60), (4, 54), (4, 48),
        (4, 42), (5, 58), (5, 50),
        (2, 58), (3, 46), (4, 38),
        (5, 42), (3, 36), (4, 32), (2, 48),
    )

    for trial, (remove_count, percentile) in enumerate(kick_schedule):
        if len(archive) == 1 or rng.random() < 0.54:
            base = archive[0][1]
        else:
            rank_limit = min(len(archive), 5)
            base = archive[int(rng.integers(rank_limit))][1]

        current = base.copy()
        present = np.flatnonzero(current[iu, ju])

        if present.size < remove_count:
            continue

        scores = obstruction_scores(current, present)
        cutoff = np.percentile(scores, percentile)
        pool = present[scores >= cutoff]

        if pool.size < remove_count:
            pool = present

        selected = []
        used = set()

        for pick in range(remove_count):
            choices = pool[~np.isin(pool, selected)]
            if choices.size == 0:
                choices = present[~np.isin(present, selected)]

            # Usually spread deletions across independent regions; periodic
            # concentrated moves can dismantle a single obstructive gadget.
            if pick > 0 and trial % 4 != 0:
                disjoint = np.array(
                    [
                        e for e in choices
                        if int(iu[e]) not in used and int(ju[e]) not in used
                    ],
                    dtype=np.int64,
                )
                if disjoint.size:
                    choices = disjoint

            chosen = int(choices[int(rng.integers(choices.size))])
            selected.append(chosen)
            used.add(int(iu[chosen]))
            used.add(int(ju[chosen]))

        banned = np.zeros(pair_count, dtype=bool)

        for e in selected:
            u = int(iu[e])
            v = int(ju[e])
            current[u, v] = 0
            current[v, u] = 0
            banned[e] = True

        if not valid(current):
            continue

        candidate = saturate(current, banned=banned)
        if valid(candidate):
            archive = archive_insert(archive, candidate)

    best = archive[0][1].copy()

    # Explicit final verification and deletion-only emergency repair.
    repair_budget = n * n

    for _ in range(repair_budget):
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

        degree = best.sum(axis=1)
        x = int(shared[np.argmax(degree[shared])])

        if degree[u] >= degree[v]:
            best[u, x] = 0
            best[x, u] = 0
        else:
            best[v, x] = 0
            best[x, v] = 0

    if valid(best):
        best = saturate(best)

    if not valid(best):
        for _ in range(repair_budget):
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
