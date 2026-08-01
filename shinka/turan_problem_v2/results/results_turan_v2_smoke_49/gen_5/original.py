# EVOLVE-BLOCK-START
"""Input-dependent local search for dense C4-free graphs."""

import numpy as np


def construct_new_graph(A, rng=None):
    """
    Delete conflicting input edges, add valid edges, and use a few bounded
    delete/refill moves to improve the resulting C4-free graph.
    """

    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]

    # Work with an undirected simple version of the supplied graph.  This is
    # still an edit of A: every retained edge originated in the input.
    original = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(original, 0)

    iu, ju = np.triu_indices(n, 1)
    input_mask = original[iu, ju].astype(bool)
    input_edges = np.flatnonzero(input_mask)
    original_degree = original.sum(axis=1).astype(np.int32)

    def valid(B):
        """Explicit C4-free verification by the common-neighbor criterion."""
        common = B.astype(np.int32) @ B.astype(np.int32)
        np.fill_diagonal(common, 0)
        return not np.any(common > 1)

    def legal_pairs(B):
        """
        A missing uv can be inserted iff there is no length-three u--v path.
        Such a path would become a 4-cycle after insertion.
        """
        X = B.astype(np.int32)
        paths3 = (X @ X) @ X
        missing = (B == 0)
        np.fill_diagonal(missing, False)
        return missing & (paths3 == 0)

    def saturate(B):
        """
        Repeatedly add a legal edge.  Among legal choices prefer endpoints
        of low current degree, which avoids early hubs and tends to produce
        the near-regular degree profiles useful for this problem.
        """
        B = B.copy()
        max_steps = n * (n - 1) // 2

        for _ in range(max_steps):
            can_add = legal_pairs(B)
            candidates = np.flatnonzero(can_add[iu, ju])
            if candidates.size == 0:
                break

            deg = B.sum(axis=1).astype(np.int32)
            scores = deg[iu[candidates]] + deg[ju[candidates]]
            best_score = scores.min()
            tied = candidates[scores == best_score]

            # Random tie breaking gives different but reproducible searches
            # under the evaluator's supplied RNG.
            chosen = tied[int(rng.integers(tied.size))]
            u, v = int(iu[chosen]), int(ju[chosen])
            B[u, v] = 1
            B[v, u] = 1

        return B

    def repair_from_input(order_kind):
        """
        Construct a C4-free subgraph solely by retaining input edges in an
        input-dependent order, then complete it by safe additions.
        """
        B = np.zeros((n, n), dtype=np.uint8)

        if input_edges.size:
            endpoint_score = (
                original_degree[iu[input_edges]] +
                original_degree[ju[input_edges]]
            ).astype(np.int32)

            noise = rng.random(input_edges.size)
            if order_kind == 0:
                # Preserve edges around less overloaded input vertices first.
                order = input_edges[np.lexsort((noise, endpoint_score))]
            elif order_kind == 1:
                # A complementary order is useful on graphs with structure.
                order = input_edges[np.lexsort((noise, -endpoint_score))]
            else:
                order = input_edges[rng.permutation(input_edges.size)]

            # Since B is always C4-free, testing before every retained edge
            # preserves the invariant throughout the repair process.
            for e in order:
                u, v = int(iu[e]), int(ju[e])
                if B[u, v]:
                    continue
                nbr_u = np.flatnonzero(B[u])
                nbr_v = np.flatnonzero(B[v])
                if nbr_u.size == 0 or nbr_v.size == 0:
                    B[u, v] = 1
                    B[v, u] = 1
                elif not np.any(B[np.ix_(nbr_u, nbr_v)]):
                    B[u, v] = 1
                    B[v, u] = 1

        return saturate(B)

    # Multiple repair orders are cheap at n=49 and make performance much less
    # sensitive to accidental ordering in the supplied matrix.
    best = None
    best_edges = -1
    for kind in range(3):
        candidate = repair_from_input(kind)
        if not valid(candidate):
            continue
        count = int(candidate.sum() // 2)
        if count > best_edges:
            best = candidate
            best_edges = count

    if best is None:
        best = np.zeros((n, n), dtype=np.uint8)
        best = saturate(best)
        best_edges = int(best.sum() // 2)

    # Small deletion/refill moves escape saturated one-edge local optima.
    # The first moves target expensive high-degree edges; later moves add
    # random diversity while remaining entirely input-state dependent.
    for trial in range(5):
        trial_graph = best.copy()
        edges = np.flatnonzero(trial_graph[iu, ju])
        if edges.size == 0:
            break

        deg = trial_graph.sum(axis=1).astype(np.int32)
        edge_score = deg[iu[edges]] + deg[ju[edges]]

        if trial < 3:
            pool = edges[edge_score >= np.percentile(edge_score, 70)]
            if pool.size == 0:
                pool = edges
        else:
            pool = edges

        remove_count = 2 if edges.size > 2 else 1
        selected = []
        available = pool.copy()
        for _ in range(remove_count):
            if available.size == 0:
                available = edges.copy()
            pick_pos = int(rng.integers(available.size))
            chosen = int(available[pick_pos])
            selected.append(chosen)
            available = available[available != chosen]

        for e in selected:
            u, v = int(iu[e]), int(ju[e])
            trial_graph[u, v] = 0
            trial_graph[v, u] = 0

        trial_graph = saturate(trial_graph)

        if valid(trial_graph):
            count = int(trial_graph.sum() // 2)
            if count > best_edges:
                best = trial_graph
                best_edges = count

    # Conservative final verification/repair.  It should normally do nothing,
    # but guarantees validity even if future modifications alter search logic.
    repair_limit = n * n
    for _ in range(repair_limit):
        common = best.astype(np.int32) @ best.astype(np.int32)
        np.fill_diagonal(common, 0)
        bad = np.argwhere(np.triu(common > 1, 1))
        if bad.size == 0:
            break

        u, v = map(int, bad[0])
        shared = np.flatnonzero(best[u] & best[v])
        if shared.size < 2:
            continue

        # Delete one incident edge from the more highly connected shared
        # neighbor; deletion cannot create a new C4.
        x = int(shared[np.argmax(best.sum(axis=1)[shared])])
        if best[u].sum() >= best[v].sum():
            best[u, x] = 0
            best[x, u] = 0
        else:
            best[v, x] = 0
            best[x, v] = 0

    # Refill after any conservative repair and verify the invariant once more.
    best = saturate(best)
    if not valid(best):
        # Extremely defensive fallback: remove one edge from each remaining
        # offending common-neighbor pair until the invariant is restored.
        for _ in range(repair_limit):
            common = best.astype(np.int32) @ best.astype(np.int32)
            np.fill_diagonal(common, 0)
            bad = np.argwhere(np.triu(common > 1, 1))
            if bad.size == 0:
                break
            u, v = map(int, bad[0])
            x = int(np.flatnonzero(best[u] & best[v])[0])
            best[u, x] = 0
            best[x, u] = 0

    np.fill_diagonal(best, 0)
    return best.astype(np.uint8)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
