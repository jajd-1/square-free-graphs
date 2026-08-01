# EVOLVE-BLOCK-START
"""Archive-based input-aware search for dense C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Improve an arbitrary adjacency matrix by deleting conflicting edges and
    adding safe edges until a dense C4-free graph is obtained.

    Search architecture:
      * construct several C4-free repairs retaining input relationships;
      * saturate each repair using future-blocking and degree-balance costs;
      * retain a small archive of good, structurally distinct states;
      * apply bounded destructive kicks and refill from archive states.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    pair_count = iu.size

    # The supplied matrix remains the source state.  Arbitrary directed or
    # asymmetric input is interpreted as an undirected simple edit target.
    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)

    source_upper = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int16)
    input_edges = np.flatnonzero(source_upper)

    def edge_count(B):
        return int(B.sum() // 2)

    def valid(B):
        """C4-free iff no off-diagonal pair has two common neighbors."""
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def insertion_legal(B, u, v):
        """
        Inserting uv is safe exactly when no existing edge joins N(u) to N(v).
        Such an edge would form a length-three u-to-v path and hence a C4.
        """
        nu = np.flatnonzero(B[u])
        nv = np.flatnonzero(B[v])
        if nu.size == 0 or nv.size == 0:
            return True
        return not np.any(B[np.ix_(nu, nv)])

    def legal_pairs(B):
        """
        All missing pairs having no length-three path between their endpoints.
        This matrix is symmetric for an undirected graph.
        """
        X = B.astype(np.int16)
        paths3 = (X @ X) @ X
        legal = (B == 0) & (paths3 == 0)
        np.fill_diagonal(legal, False)
        return legal

    def saturate(B, banned=None):
        """
        Safely complete a C4-free graph.

        The primary score estimates legal pairs destroyed by adding uv.
        The secondary degree term favors the near-regular profiles expected in
        dense C4-free graphs.  Temporarily banned deleted edges force kicks to
        seek replacements before restoring their old structure.
        """
        B = B.copy()

        if banned is None:
            blocked_edges = np.zeros(pair_count, dtype=bool)
        else:
            blocked_edges = banned.copy()

        for _ in range(pair_count):
            legal = legal_pairs(B)
            legal_upper = legal[iu, ju]
            available = np.flatnonzero(legal_upper & ~blocked_edges)

            # A kicked-out edge may be restored only after alternatives have
            # been exhausted, ensuring returned graphs are fully saturated.
            if available.size == 0 and np.any(blocked_edges):
                blocked_edges[:] = False
                available = np.flatnonzero(legal_upper)

            if available.size == 0:
                break

            X = B.astype(np.int16)
            L = legal.astype(np.int16)

            # Number of legal cross-pairs x-y with x in N(u), y in N(v).
            # Each such pair becomes unavailable after uv is inserted.
            future_loss = (X @ L) @ X
            degree = B.sum(axis=1).astype(np.int16)

            au = iu[available]
            av = ju[available]
            loss = future_loss[au, av].astype(np.int32)
            load = (
                degree[au].astype(np.int32) +
                degree[av].astype(np.int32)
            )

            score = 3 * loss + 2 * load
            minimum = score.min()
            tied = available[score == minimum]

            # Input ties preserve dependence on A but never override the
            # structural score, avoiding commitment to poor source patterns.
            source_tied = tied[source_upper[tied]]
            if source_tied.size and rng.random() < 0.60:
                tied = source_tied

            chosen = int(tied[int(rng.integers(tied.size))])
            u, v = int(iu[chosen]), int(ju[chosen])

            # Explicit local invariant protection.
            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def repaired_input_state(style):
        """
        Retain a maximal C4-free subset of input edges in one of several
        source-dependent orders.  Every retained edge is directly inherited
        from the supplied matrix before augmentation begins.
        """
        B = np.zeros((n, n), dtype=np.uint8)

        if input_edges.size == 0:
            return B

        du = source_degree[iu[input_edges]].astype(np.int32)
        dv = source_degree[ju[input_edges]].astype(np.int32)
        load = du + dv
        product = du * dv
        noise = rng.random(input_edges.size)

        if style == 0:
            order = input_edges[np.lexsort((noise, load))]
        elif style == 1:
            order = input_edges[np.lexsort((noise, -load))]
        elif style == 2:
            order = input_edges[np.lexsort((noise, product))]
        elif style == 3:
            order = input_edges[np.lexsort((noise, -product))]
        else:
            order = input_edges[rng.permutation(input_edges.size)]

        for edge in order:
            u, v = int(iu[edge]), int(ju[edge])
            if insertion_legal(B, u, v):
                B[u, v] = 1
                B[v, u] = 1

        return B

    def archive_insert(archive, candidate, limit=4):
        """
        Keep a compact elite archive.  Exact duplicate states are discarded;
        retaining several equal-score states gives later kicks useful variety.
        """
        if not valid(candidate):
            return archive

        count = edge_count(candidate)
        signature = candidate[iu, ju].tobytes()

        for old_count, old_graph, old_signature in archive:
            if signature == old_signature:
                return archive

        archive.append((count, candidate.copy(), signature))
        archive.sort(key=lambda item: item[0], reverse=True)
        return archive[:limit]

    # Build source-derived initial states.  There are complementary ordered
    # repairs plus two independently randomized repairs.
    archive = []
    for style in range(6):
        state = repaired_input_state(style)
        if not valid(state):
            continue

        state = saturate(state)
        if valid(state):
            archive = archive_insert(archive, state)

    # Defensive fallback still begins from the input-derived empty repair.
    if not archive:
        fallback = saturate(np.zeros((n, n), dtype=np.uint8))
        archive = archive_insert(archive, fallback)

    # Kicks use several archive states rather than only the current champion.
    # This avoids repeatedly exploring the same saturated basin.
    kick_schedule = (
        (2, 72), (2, 72), (2, 65), (2, 65),
        (3, 60), (3, 60), (3, 55), (3, 55),
        (3, 48), (4, 48), (4, 42), (4, 35),
        (2, 55), (3, 50), (4, 40),
    )

    for trial, (remove_count, percentile) in enumerate(kick_schedule):
        if len(archive) == 1 or rng.random() < 0.62:
            base = archive[0][1]
        else:
            # Favor stronger archive entries while still allowing alternate
            # local optima to seed destructive/refill exploration.
            index = int(rng.integers(len(archive)))
            base = archive[index][1]

        current = base.copy()
        present = np.flatnonzero(current[iu, ju])
        if present.size < remove_count:
            continue

        degree = current.sum(axis=1).astype(np.int16)
        pressure = (
            degree[iu[present]].astype(np.int32) +
            degree[ju[present]].astype(np.int32)
        )

        threshold = np.percentile(pressure, percentile)
        pool = present[pressure >= threshold]
        if pool.size < remove_count:
            pool = present

        # Select expensive edges, usually spreading deletions over endpoints.
        # Distributed destruction tends to unlock more independent additions.
        selected = []
        used_vertices = set()

        for pick_number in range(remove_count):
            choices = pool[~np.isin(pool, selected)]
            if choices.size == 0:
                choices = present[~np.isin(present, selected)]

            if pick_number > 0 and trial % 3 != 0:
                disjoint = np.array(
                    [
                        e for e in choices
                        if int(iu[e]) not in used_vertices
                        and int(ju[e]) not in used_vertices
                    ],
                    dtype=np.int64,
                )
                if disjoint.size:
                    choices = disjoint

            chosen = int(choices[int(rng.integers(choices.size))])
            selected.append(chosen)
            used_vertices.add(int(iu[chosen]))
            used_vertices.add(int(ju[chosen]))

        banned = np.zeros(pair_count, dtype=bool)
        for edge in selected:
            u, v = int(iu[edge]), int(ju[edge])
            current[u, v] = 0
            current[v, u] = 0
            banned[edge] = True

        # Deletion is monotone-safe, but validate before entering refill.
        if not valid(current):
            continue

        candidate = saturate(current, banned)
        if valid(candidate):
            archive = archive_insert(archive, candidate)

    best = archive[0][1].copy()

    # Explicit final verification and conservative monotone repair.
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
        if shared.size < 2:
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

    # Strict deletion-only fallback if an unexpected implementation error ever
    # leaves a violation.  Deletions cannot create new C4s.
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
