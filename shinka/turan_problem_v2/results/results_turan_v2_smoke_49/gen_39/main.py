# EVOLVE-BLOCK-START
"""Bitset balanced-ejection search for dense input-derived C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Edit A into a dense C4-free graph.

    This implementation uses integer neighborhood masks rather than repeated
    matrix products during construction.  For an absent edge uv, insertion is
    safe exactly when there is no current edge between N(u) and N(v).
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    m = len(iu)

    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)
    source_upper = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int32)
    source_edges = np.flatnonzero(source_upper)

    edge_id = -np.ones((n, n), dtype=np.int32)
    edge_id[iu, ju] = np.arange(m, dtype=np.int32)
    edge_id[ju, iu] = np.arange(m, dtype=np.int32)

    def matrix_valid(B):
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def edge_total(masks):
        return sum(x.bit_count() for x in masks) // 2

    def add_edge(masks, deg, u, v):
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(masks, deg, u, v):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1

    def legal_insert(masks, u, v):
        """
        Check that no edge crosses N(u), N(v).  This is precisely the
        no-four-cycle insertion criterion for a currently C4-free graph.
        """
        nu = masks[u]
        nv = masks[v]

        # Iterating through the smaller neighborhood is substantially faster.
        if nu.bit_count() > nv.bit_count():
            nu, nv = nv, nu

        while nu:
            low = nu & -nu
            w = low.bit_length() - 1
            if masks[w] & nv:
                return False
            nu ^= low
        return True

    def masks_to_matrix(masks):
        B = np.zeros((n, n), dtype=np.uint8)
        for u in range(n):
            row = masks[u]
            while row:
                low = row & -row
                v = low.bit_length() - 1
                B[u, v] = 1
                row ^= low
        np.fill_diagonal(B, 0)
        return B

    def complete(masks, deg, banned=None, source_bias=0.34, style=0):
        """
        Saturate a valid graph.  At every iteration all absent pairs are
        considered, giving an explicit complete saturation pass.

        The primary criterion favors low endpoint degree, which encourages the
        near-regular degree profile characteristic of dense C4-free graphs.
        A mild high-degree penalty and a randomized narrow tie band preserve
        diversity between otherwise equivalent completions.
        """
        masks = masks.copy()
        deg = deg.copy()
        forbidden = np.zeros(m, dtype=bool) if banned is None else banned.copy()

        for _ in range(m):
            best_score = None
            candidates = []

            for e in range(m):
                if forbidden[e]:
                    continue

                u = int(iu[e])
                v = int(ju[e])

                if (masks[u] >> v) & 1:
                    continue
                if not legal_insert(masks, u, v):
                    continue

                du = int(deg[u])
                dv = int(deg[v])
                overload = max(du - 7, 0) + max(dv - 7, 0)

                if style == 0:
                    score = 5 * (du + dv) + 4 * overload
                elif style == 1:
                    score = 4 * (du + dv) + 7 * overload + abs(du - dv)
                else:
                    score = 5 * (du + dv) + 8 * overload - min(du, dv)

                if best_score is None or score < best_score:
                    best_score = score
                    candidates = [e]
                elif score <= best_score + 2:
                    candidates.append(e)

            if not candidates:
                # During ejection/refill, force alternatives before allowing
                # immediately deleted edges to return.
                if np.any(forbidden):
                    forbidden[:] = False
                    continue
                break

            cand = np.asarray(candidates, dtype=np.int32)
            original = cand[source_upper[cand]]
            if original.size and rng.random() < source_bias:
                cand = original

            chosen = int(cand[int(rng.integers(cand.size))])
            u = int(iu[chosen])
            v = int(ju[chosen])

            # Redundant local check keeps the invariant explicit.
            if legal_insert(masks, u, v):
                add_edge(masks, deg, u, v)

        return masks, deg

    def source_scaffold(style, prefix):
        """
        Build a C4-free subset containing only edges actually supplied in A.
        Different source orderings lead to distinct structural basins.
        """
        masks = [0] * n
        deg = np.zeros(n, dtype=np.int16)

        if source_edges.size == 0:
            return masks, deg

        e = source_edges
        u = iu[e]
        v = ju[e]
        du = source_degree[u]
        dv = source_degree[v]
        load = du + dv
        prod = du * dv
        spread = np.abs(du - dv)
        noise = rng.random(e.size)

        if style == 0:
            order = e[np.lexsort((noise, load))]
        elif style == 1:
            order = e[np.lexsort((noise, -load))]
        elif style == 2:
            order = e[np.lexsort((noise, prod))]
        elif style == 3:
            order = e[np.lexsort((noise, -prod))]
        elif style == 4:
            order = e[np.lexsort((noise, spread))]
        elif style == 5:
            order = e[np.lexsort((noise, -spread))]
        else:
            # A random source-derived order remains a real edit of A.
            order = e[rng.permutation(e.size)]

        if prefix is not None:
            order = order[:min(int(prefix), order.size)]

        for edge in order:
            u0 = int(iu[edge])
            v0 = int(ju[edge])
            if legal_insert(masks, u0, v0):
                add_edge(masks, deg, u0, v0)

        return masks, deg

    archive = []

    def archive_add(masks, deg):
        B = masks_to_matrix(masks)
        if not matrix_valid(B):
            return

        signature = B[iu, ju].tobytes()
        for _, _, _, old_sig in archive:
            if signature == old_sig:
                return

        archive.append((edge_total(masks), masks.copy(), deg.copy(), signature))
        archive.sort(key=lambda z: z[0], reverse=True)
        del archive[8:]

    # Several initial states retain actual input edges.  Short prefixes are
    # valuable for dense inputs because they avoid inheriting one bad source
    # region too strongly.
    plans = (
        (0, None, 0), (1, None, 1), (2, None, 2), (3, None, 0),
        (4, None, 1), (5, None, 2),
        (0, 12, 1), (1, 20, 2), (2, 30, 0),
        (4, 42, 1), (6, 54, 2),
    )

    for style, prefix, completion_style in plans:
        masks, deg = source_scaffold(style, prefix)
        masks, deg = complete(
            masks, deg,
            source_bias=0.44 if prefix is None else 0.24,
            style=completion_style,
        )
        archive_add(masks, deg)

    if not archive:
        masks = [0] * n
        deg = np.zeros(n, dtype=np.int16)
        masks, deg = complete(masks, deg, source_bias=0.0, style=0)
        archive_add(masks, deg)

    def ejection_weights(masks, deg):
        """
        Rank edges by pressure at their endpoints.  Removing edges adjacent to
        overloaded vertices is an inexpensive proxy for destroying many
        length-three obstructions while improving degree balance.
        """
        weights = np.full(m, -10**6, dtype=np.int32)
        for e in range(m):
            u = int(iu[e])
            v = int(ju[e])
            if (masks[u] >> v) & 1:
                du = int(deg[u])
                dv = int(deg[v])
                weights[e] = (
                    8 * max(du - 6, 0)
                    + 8 * max(dv - 6, 0)
                    + du + dv
                    + int(rng.integers(5))
                )
        return weights

    # Ejection/refill is intentionally smaller but more numerous than broad
    # destructive restarts.  Temporary bans make these true exchange moves.
    schedule = (
        (1, 82, 0), (2, 78, 1), (2, 64, 2),
        (3, 74, 0), (3, 60, 1), (3, 46, 2),
        (4, 70, 0), (4, 56, 1), (4, 40, 2),
        (5, 62, 0), (5, 48, 1), (6, 44, 2),
        (2, 32, 0), (3, 30, 1), (4, 28, 2),
        (5, 34, 0),
    )

    for trial, (remove_count, percentile, style) in enumerate(schedule):
        if len(archive) == 1 or rng.random() < 0.63:
            _, base_masks, base_deg, _ = archive[0]
        else:
            rank = int(rng.integers(min(5, len(archive))))
            _, base_masks, base_deg, _ = archive[rank]

        masks = base_masks.copy()
        deg = base_deg.copy()

        present = np.array(
            [e for e in range(m) if (masks[int(iu[e])] >> int(ju[e])) & 1],
            dtype=np.int32,
        )
        if present.size < remove_count:
            continue

        weights = ejection_weights(masks, deg)
        threshold = np.percentile(weights[present], percentile)
        pool = present[weights[present] >= threshold]
        if pool.size < remove_count:
            pool = present

        chosen = []
        used = set()

        for pick in range(remove_count):
            options = np.array(
                [e for e in pool if int(e) not in chosen],
                dtype=np.int32,
            )
            if options.size == 0:
                options = np.array(
                    [e for e in present if int(e) not in chosen],
                    dtype=np.int32,
                )

            # Except for designated concentrated kicks, spread deletions over
            # separate neighborhoods.
            if pick > 0 and trial % 4 != 0:
                independent = np.array(
                    [
                        e for e in options
                        if int(iu[e]) not in used and int(ju[e]) not in used
                    ],
                    dtype=np.int32,
                )
                if independent.size:
                    options = independent

            e = int(options[int(rng.integers(options.size))])
            chosen.append(e)
            used.add(int(iu[e]))
            used.add(int(ju[e]))

        banned = np.zeros(m, dtype=bool)
        for e in chosen:
            u = int(iu[e])
            v = int(ju[e])
            remove_edge(masks, deg, u, v)
            banned[e] = True

        candidate_masks, candidate_deg = complete(
            masks,
            deg,
            banned=banned,
            source_bias=0.18,
            style=style,
        )
        archive_add(candidate_masks, candidate_deg)

    best = archive[0][1].copy()
    B = masks_to_matrix(best)

    # Defensive explicit invariant verification and deletion-only repair.
    repair_budget = n * n
    for _ in range(repair_budget):
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        bad = np.argwhere(np.triu(common > 1, 1))
        if bad.size == 0:
            break

        u, v = map(int, bad[0])
        shared = np.flatnonzero(B[u] & B[v])
        if shared.size == 0:
            break

        degree = B.sum(axis=1)
        w = int(shared[np.argmax(degree[shared])])
        if degree[u] >= degree[v]:
            B[u, w] = 0
            B[w, u] = 0
        else:
            B[v, w] = 0
            B[w, v] = 0

    # Rebuild masks after any defensive repair and explicitly saturate again.
    if matrix_valid(B):
        masks = [0] * n
        deg = B.sum(axis=1).astype(np.int16)
        for u in range(n):
            row = 0
            for v in np.flatnonzero(B[u]):
                row |= (1 << int(v))
            masks[u] = row
        masks, deg = complete(masks, deg, source_bias=0.18, style=0)
        B = masks_to_matrix(masks)

    # Absolute last-resort validity repair.
    for _ in range(repair_budget):
        if matrix_valid(B):
            break
        X = B.astype(np.int16)
        common = X @ X
        np.fill_diagonal(common, 0)
        bad = np.argwhere(np.triu(common > 1, 1))
        if bad.size == 0:
            break
        u, v = map(int, bad[0])
        shared = np.flatnonzero(B[u] & B[v])
        if shared.size == 0:
            break
        w = int(shared[0])
        B[u, w] = 0
        B[w, u] = 0

    np.fill_diagonal(B, 0)
    return B.astype(np.uint8)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
