# EVOLVE-BLOCK-START
"""Conflict-credit exchange search for dense input-aware C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Transform an arbitrary adjacency matrix into a dense C4-free graph.

    The algorithm retains source edges in several source-dependent orders,
    saturates safely using future-blocking costs, and performs bounded
    exchange moves that remove edges with high conflict credit before refill.
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
    source_degree = source.sum(axis=1).astype(np.int16)
    source_edges = np.flatnonzero(source_upper)

    def edge_count(B):
        return int(B.sum() // 2)

    def valid(B):
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def insertion_legal(B, u, v):
        nu = np.flatnonzero(B[u])
        nv = np.flatnonzero(B[v])
        if nu.size == 0 or nv.size == 0:
            return True
        return not np.any(B[np.ix_(nu, nv)])

    def legal_pairs(B):
        X = B.astype(np.int16)
        paths3 = (X @ X) @ X
        legal = (B == 0) & (paths3 == 0)
        np.fill_diagonal(legal, False)
        return legal

    def saturate(B, banned_edges=None):
        """
        Add safe edges to maximality.  The main criterion is the exact number
        of currently legal pairs destroyed by an insertion.  A quadratic
        degree penalty discourages premature hubs more strongly than a simple
        endpoint-load score.
        """
        B = B.copy()

        if banned_edges is None:
            banned = np.zeros(pair_count, dtype=bool)
        else:
            banned = banned_edges.copy()

        for _ in range(pair_count):
            legal = legal_pairs(B)
            available = np.flatnonzero(legal[iu, ju] & ~banned)

            if available.size == 0 and np.any(banned):
                banned[:] = False
                available = np.flatnonzero(legal[iu, ju])

            if available.size == 0:
                break

            X = B.astype(np.int16)
            L = legal.astype(np.int16)
            blocked = (X @ L) @ X
            degree = B.sum(axis=1).astype(np.int32)

            au = iu[available]
            av = ju[available]

            future_loss = blocked[au, av].astype(np.int32)
            endpoint_sum = degree[au] + degree[av]

            # Dense C4-free graphs on 49 vertices generally benefit from
            # degrees concentrated around 6--7.  Penalize excessive imbalance.
            imbalance = (
                np.maximum(degree[au] - 7, 0) ** 2
                + np.maximum(degree[av] - 7, 0) ** 2
            )

            score = 3 * future_loss + 2 * endpoint_sum + 3 * imbalance
            best_score = score.min()
            tied = available[score == best_score]

            source_tied = tied[source_upper[tied]]
            if source_tied.size and rng.random() < 0.58:
                tied = source_tied

            chosen = int(tied[int(rng.integers(tied.size))])
            u = int(iu[chosen])
            v = int(ju[chosen])

            # Redundant local check protects the invariant even if the
            # matrix-based calculation is modified in future versions.
            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def source_repair(style):
        """
        Construct a C4-free retained subgraph of the supplied input graph.
        Different orderings deliberately preserve different source structure.
        """
        B = np.zeros((n, n), dtype=np.uint8)

        if source_edges.size == 0:
            return B

        du = source_degree[iu[source_edges]].astype(np.int32)
        dv = source_degree[ju[source_edges]].astype(np.int32)
        load = du + dv
        product = du * dv
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
            # A random repair remains source-derived but avoids deterministic
            # preference for low-index vertices.
            order = source_edges[rng.permutation(source_edges.size)]

        for edge in order:
            u = int(iu[edge])
            v = int(ju[edge])
            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def archive_insert(archive, B, limit=4):
        """Store good, nonduplicate C4-free local optima."""
        if not valid(B):
            return archive

        signature = B[iu, ju].tobytes()
        for _, _, old_signature in archive:
            if signature == old_signature:
                return archive

        archive.append((edge_count(B), B.copy(), signature))
        archive.sort(key=lambda item: item[0], reverse=True)
        return archive[:limit]

    archive = []

    # Several source-derived starts.  All augmentation happens after retaining
    # a valid subset of A, so the supplied graph directly affects search.
    for style in range(5):
        candidate = source_repair(style)
        if valid(candidate):
            candidate = saturate(candidate)
            archive = archive_insert(archive, candidate)

    if not archive:
        fallback = saturate(np.zeros((n, n), dtype=np.uint8))
        archive = archive_insert(archive, fallback)

    # Exchange schedule: early moves are conservative; later moves remove
    # more conflict-heavy edges to cross larger local-optimum barriers.
    schedule = (
        (2, 78), (2, 72), (2, 68), (2, 62),
        (3, 72), (3, 66), (3, 60), (3, 54),
        (4, 65), (4, 57), (4, 48), (3, 45),
        (2, 55), (4, 40),
    )

    for trial, (remove_count, percentile) in enumerate(schedule):
        if len(archive) == 1 or rng.random() < 0.65:
            base = archive[0][1]
        else:
            base = archive[int(rng.integers(len(archive)))][1]

        current = base.copy()
        present = np.flatnonzero(current[iu, ju])

        if present.size < remove_count:
            continue

        degree = current.sum(axis=1).astype(np.int32)
        pu = iu[present]
        pv = ju[present]

        # Conflict credit estimates how many endpoint-crossing paths use an
        # edge as their middle link.  Removing high-credit edges tends to
        # unlock more candidate replacement edges than degree sum alone.
        credit = (
            np.maximum(degree[pu] - 1, 0)
            * np.maximum(degree[pv] - 1, 0)
        )
        pressure = degree[pu] + degree[pv]
        combined = 4 * credit + pressure

        threshold = np.percentile(combined, percentile)
        pool = present[combined >= threshold]
        if pool.size < remove_count:
            pool = present

        selected = []
        used = set()

        for pick in range(remove_count):
            choices = pool[~np.isin(pool, selected)]
            if choices.size == 0:
                choices = present[~np.isin(present, selected)]

            # Favor distributed deletion: independent removed edges usually
            # release separate regions of the saturation constraint system.
            if pick > 0 and trial % 3 != 0:
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
        for edge in selected:
            u = int(iu[edge])
            v = int(ju[edge])
            current[u, v] = 0
            current[v, u] = 0
            banned[edge] = True

        if not valid(current):
            continue

        candidate = saturate(current, banned)

        if valid(candidate):
            archive = archive_insert(archive, candidate)

    best = archive[0][1].copy()

    # Conservative explicit final verification.  This is normally already
    # valid, but deletion-only repair guarantees a valid returned graph.
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

        degrees = best.sum(axis=1)
        x = int(shared[np.argmax(degrees[shared])])

        if degrees[u] >= degrees[v]:
            best[u, x] = 0
            best[x, u] = 0
        else:
            best[v, x] = 0
            best[x, v] = 0

    if valid(best):
        best = saturate(best)

    # Strict fallback if a future alteration ever creates a violation.
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
            if shared.size == 0:
                continue

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
