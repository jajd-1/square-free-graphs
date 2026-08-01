# EVOLVE-BLOCK-START
import numpy as np


def construct_new_graph(A, rng=None):
    """
    Input-aware dense C4-free graph search.

    The search starts from several repaired / source-preserving subgraphs of
    A, greedily completes each one using safe bitset edge insertions, and then
    improves an archive of maximal graphs through tabu ruin-and-refill moves.
    """
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    iu, ju = np.triu_indices(n, 1)
    pair_count = len(iu)

    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)
    source_upper = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int16)
    source_edges = np.flatnonzero(source_upper)

    INITIAL_REPAIRS = 6
    INITIAL_GREEDY = 5
    ARCHIVE_LIMIT = 6

    def common_counts(H):
        X = H.astype(np.int16, copy=False)
        return X @ X

    def valid(H):
        C = common_counts(H)
        np.fill_diagonal(C, 0)
        return not np.any(C > 1)

    def matrix_to_masks(H):
        masks = [0] * n
        for u in range(n):
            bits = 0
            for v in np.flatnonzero(H[u]):
                bits |= 1 << int(v)
            masks[u] = bits
        return masks

    def masks_to_matrix(masks):
        H = np.zeros((n, n), dtype=np.uint8)
        for u, bits in enumerate(masks):
            x = bits
            while x:
                low = x & -x
                v = low.bit_length() - 1
                H[u, v] = 1
                x ^= low
        np.fill_diagonal(H, 0)
        return H

    def edge_count(masks):
        return sum(x.bit_count() for x in masks) // 2

    def legal_add(masks, u, v):
        """
        uv is legal iff there is no existing edge from N(u) to N(v).
        This is equivalent to saying that adding uv cannot close a C4.
        """
        nv = masks[v]
        x = masks[u]
        while x:
            low = x & -x
            w = low.bit_length() - 1
            if masks[w] & nv:
                return False
            x ^= low
        return True

    def repair(H, style):
        """
        Conflict-directed deletion repair.  Every deletion is monotone with
        respect to validity, and the score selects edges covering many active
        common-neighbor violations.
        """
        H = H.copy()
        limit = pair_count

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
                deg = H.sum(axis=1).astype(np.int16)

                if style % 4 == 0:
                    weight = deg[choices[:, 0]] + deg[choices[:, 1]]
                    choices = choices[weight == weight.max()]
                elif style % 4 == 1:
                    weight = deg[choices[:, 0]] * deg[choices[:, 1]]
                    choices = choices[weight == weight.max()]
                elif style % 4 == 2:
                    weight = np.maximum(
                        deg[choices[:, 0]], deg[choices[:, 1]]
                    )
                    choices = choices[weight == weight.max()]
                else:
                    weight = np.abs(
                        deg[choices[:, 0]] - deg[choices[:, 1]]
                    )
                    choices = choices[weight == weight.max()]

                u, v = choices[int(rng.integers(len(choices)))]
                u, v = int(u), int(v)
            else:
                pairs = np.argwhere(np.triu(bad, 1))
                a, b = pairs[int(rng.integers(len(pairs)))]
                shared = np.flatnonzero(H[a] & H[b])
                w = int(shared[int(rng.integers(len(shared)))])
                if H[a].sum() >= H[b].sum():
                    u, v = int(a), w
                else:
                    u, v = int(b), w

            H[u, v] = 0
            H[v, u] = 0

        # Conservative deletion-only completion safeguard.
        for _ in range(limit):
            C = common_counts(H)
            bad_pairs = np.argwhere(np.triu(C > 1, 1))
            if bad_pairs.size == 0:
                break
            a, b = map(int, bad_pairs[0])
            shared = np.flatnonzero(H[a] & H[b])
            if shared.size == 0:
                continue
            w = int(shared[0])
            if H[a].sum() >= H[b].sum():
                H[a, w] = H[w, a] = 0
            else:
                H[b, w] = H[w, b] = 0

        return H

    def source_greedy(style):
        """
        Construct a C4-free subgraph by retaining input edges in several
        genuinely input-dependent orders.
        """
        masks = [0] * n
        degrees = [0] * n

        if source_edges.size == 0:
            return masks, degrees

        a = iu[source_edges]
        b = ju[source_edges]
        load = (
            source_degree[a].astype(np.int32)
            + source_degree[b].astype(np.int32)
        )
        product = (
            source_degree[a].astype(np.int32)
            * source_degree[b].astype(np.int32)
        )
        imbalance = np.abs(
            source_degree[a].astype(np.int32)
            - source_degree[b].astype(np.int32)
        )
        noise = rng.random(source_edges.size)

        if style == 0:
            order = source_edges[np.lexsort((noise, load))]
        elif style == 1:
            order = source_edges[np.lexsort((noise, product))]
        elif style == 2:
            order = source_edges[np.lexsort((noise, imbalance))]
        elif style == 3:
            order = source_edges[np.lexsort((noise, -load))]
        else:
            order = source_edges[rng.permutation(source_edges.size)]

        for e in order:
            u, v = int(iu[e]), int(ju[e])
            if legal_add(masks, u, v):
                masks[u] |= 1 << v
                masks[v] |= 1 << u
                degrees[u] += 1
                degrees[v] += 1

        return masks, degrees

    def saturate(masks, degrees, mode=0, taboo=None):
        """
        Safely complete a valid graph to maximality.

        mode 0: strongest regularity pressure.
        mode 1: product-first selection, preserving future pair budget.
        mode 2: mixed selection with a mild preference for supplied edges.
        """
        masks = masks[:]
        degrees = degrees[:]
        taboo = set() if taboo is None else set(taboo)
        taboo_active = bool(taboo)

        for _ in range(pair_count):
            best_key = None
            candidates = []

            for u in range(n - 1):
                mu = masks[u]
                du = degrees[u]

                for v in range(u + 1, n):
                    if (mu >> v) & 1:
                        continue

                    edge_id = u * n + v
                    if taboo_active and edge_id in taboo:
                        continue

                    if not legal_add(masks, u, v):
                        continue

                    dv = degrees[v]
                    high = max(du, dv)
                    total = du + dv
                    product = du * dv
                    imbalance = abs(du - dv)
                    excess = max(0, high - 7)

                    if mode % 3 == 0:
                        key = (total, high, product, imbalance)
                    elif mode % 3 == 1:
                        key = (product, total, high, imbalance)
                    else:
                        key = (
                            total + 2 * excess * excess,
                            product,
                            imbalance,
                            high,
                        )

                    if best_key is None or key < best_key:
                        best_key = key
                        candidates = [(u, v)]
                    elif key == best_key:
                        candidates.append((u, v))

            if not candidates:
                if taboo_active:
                    taboo_active = False
                    continue
                break

            # Source preference is intentionally weak: it only resolves ties.
            if mode % 3 == 2 and len(candidates) > 1 and rng.random() < 0.42:
                preferred = [
                    (u, v) for u, v in candidates
                    if source[u, v] != 0
                ]
                if preferred:
                    candidates = preferred

            u, v = candidates[int(rng.integers(len(candidates)))]
            masks[u] |= 1 << v
            masks[v] |= 1 << u
            degrees[u] += 1
            degrees[v] += 1

        return masks, degrees

    def normalize(masks, degrees, mode):
        H = masks_to_matrix(masks)
        if not valid(H):
            H = repair(H, mode)
            masks = matrix_to_masks(H)
            degrees = [x.bit_count() for x in masks]

        masks, degrees = saturate(masks, degrees, mode=mode)
        H = masks_to_matrix(masks)

        if not valid(H):
            H = repair(H, mode + 1)
            masks = matrix_to_masks(H)
            degrees = [x.bit_count() for x in masks]
            masks, degrees = saturate(masks, degrees, mode=mode)

        return masks, degrees

    def add_archive(archive, masks, degrees):
        H = masks_to_matrix(masks)
        if not valid(H):
            return archive

        sig = tuple(masks)
        for _, _, _, old_sig in archive:
            if sig == old_sig:
                return archive

        archive.append((edge_count(masks), masks[:], degrees[:], sig))
        archive.sort(key=lambda item: item[0], reverse=True)
        return archive[:ARCHIVE_LIMIT]

    def removal_candidates(masks, degrees, remove_count, broad):
        """
        An edge uv blocks cross-neighborhood pairs between N(u) and N(v).
        In a valid graph there are no current edges between these sets, so
        (deg(u)-1)(deg(v)-1) is a cheap direct unlocking estimate.
        """
        scored = []

        for u in range(n - 1):
            x = masks[u] >> (u + 1)
            v = u + 1
            while x:
                if x & 1:
                    unlock = (degrees[u] - 1) * (degrees[v] - 1)
                    pressure = degrees[u] + degrees[v]
                    balance = abs(degrees[u] - degrees[v])
                    scored.append((8 * unlock + pressure - balance, u, v))
                x >>= 1
                v += 1

        if not scored:
            return []

        scored.sort(reverse=True)
        divisor = 2 if broad else 4
        pool_size = max(remove_count + 2, len(scored) // divisor + 4)
        pool = scored[:pool_size]

        selected = []
        used = set()

        for pick in range(remove_count):
            available = [
                item for item in pool
                if (item[1], item[2]) not in [
                    (old[1], old[2]) for old in selected
                ]
            ]

            if pick > 0 and available:
                disjoint = [
                    item for item in available
                    if item[1] not in used and item[2] not in used
                ]
                if disjoint and (pick % 3 != 2):
                    available = disjoint

            if not available:
                break

            chosen = available[int(rng.integers(len(available)))]
            selected.append(chosen)
            used.add(chosen[1])
            used.add(chosen[2])

        return selected

    archive = []

    # Input-derived deletion repairs.
    for style in range(INITIAL_REPAIRS):
        repaired = repair(source, style)
        masks = matrix_to_masks(repaired)
        degrees = [x.bit_count() for x in masks]
        masks, degrees = normalize(masks, degrees, style)
        archive = add_archive(archive, masks, degrees)

    # Input-derived constructive retention starts.
    for style in range(INITIAL_GREEDY):
        masks, degrees = source_greedy(style)
        masks, degrees = normalize(masks, degrees, style + 1)
        archive = add_archive(archive, masks, degrees)

    if not archive:
        masks = [0] * n
        degrees = [0] * n
        masks, degrees = saturate(masks, degrees, mode=0)
        archive = add_archive(archive, masks, degrees)

    # (edges deleted, broad removal pool, saturation mode)
    kick_schedule = (
        (2, False, 0), (2, True, 1), (3, False, 2),
        (3, True, 0), (4, False, 1), (3, False, 2),
        (4, True, 0), (5, False, 1), (2, True, 2),
        (4, False, 0), (5, True, 1), (3, True, 2),
        (6, True, 0), (4, False, 1), (2, False, 2),
        (5, True, 0), (3, False, 1), (4, True, 2),
    )

    for round_id, (remove_count, broad, mode) in enumerate(kick_schedule):
        bases = archive[:min(4, len(archive))]
        proposals = []

        for rank, (_, base_masks, base_degrees, _) in enumerate(bases):
            attempts = 2 if rank == 0 else 1

            for attempt in range(attempts):
                masks = base_masks[:]
                degrees = base_degrees[:]

                removed = removal_candidates(
                    masks,
                    degrees,
                    remove_count,
                    broad=(broad or ((round_id + attempt) % 4 == 0)),
                )
                if not removed:
                    continue

                taboo = set()
                for _, u, v in removed:
                    if (masks[u] >> v) & 1:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        degrees[u] -= 1
                        degrees[v] -= 1
                        taboo.add(u * n + v)

                H = masks_to_matrix(masks)
                if not valid(H):
                    H = repair(H, round_id)
                    masks = matrix_to_masks(H)
                    degrees = [x.bit_count() for x in masks]

                masks, degrees = saturate(
                    masks,
                    degrees,
                    mode=(mode + attempt + rank) % 3,
                    taboo=taboo,
                )

                H = masks_to_matrix(masks)
                if valid(H):
                    proposals.append((masks, degrees))

        for masks, degrees in proposals:
            archive = add_archive(archive, masks, degrees)

    result = masks_to_matrix(archive[0][1])

    # Mandatory conservative final verification.
    if not valid(result):
        result = repair(result, 0)
        masks = matrix_to_masks(result)
        degrees = [x.bit_count() for x in masks]
        masks, degrees = saturate(masks, degrees, mode=0)
        result = masks_to_matrix(masks)

    if not valid(result):
        result = repair(result, 1)

    np.fill_diagonal(result, 0)
    return result.astype(np.uint8, copy=False)


# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
