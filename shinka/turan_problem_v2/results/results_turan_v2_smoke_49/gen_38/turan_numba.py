"""Numba-accelerated version of ShinkaEvolve Turan v2 generation 38.

The public ``construct_new_graph`` signature intentionally matches the
generated program, so it can be imported directly by Axplorer's square
environment.  The search policy, schedules, and calls to the supplied NumPy
RNG are unchanged.  Only deterministic hot paths are compiled.

This implementation targets the C4-free invariant.  In particular, a graph
assembled by safe insertions is valid, and removing edges preserves validity.
That makes the original post-scaffold, post-deletion, and post-saturation
validity checks redundant; they are intentionally omitted here.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit


ARCHIVE_SIZE = 14
PLANS = (
    (0, None), (1, None), (2, None), (3, None), (4, None), (5, None),
    (6, None), (0, 16), (1, 22), (2, 30), (3, 38), (4, 46), (5, 54),
    (7, 34), (7, 48),
)
SCHEDULE = (
    (1, 88), (1, 78), (2, 86), (2, 78), (2, 70), (2, 62),
    (3, 82), (3, 74), (3, 66), (3, 56), (3, 46),
    (4, 76), (4, 68), (4, 60), (4, 50), (4, 40),
    (5, 70), (5, 60), (5, 50), (5, 40),
    (6, 66), (6, 56), (6, 46), (7, 60), (7, 48), (8, 54), (8, 42),
    (3, 32), (4, 30), (5, 34), (6, 36),
)


@njit(cache=False)
def _is_valid(B):
    """Return whether every vertex pair has at most one common neighbour."""
    n = B.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in range(n):
                if B[u, w] != 0 and B[v, w] != 0:
                    common += 1
                    if common > 1:
                        return False
    return True


@njit(cache=False)
def _is_legal_edge(B, u, v):
    """Adding uv is safe iff no edge joins N(u) to N(v)."""
    n = B.shape[0]
    for a in range(n):
        if B[u, a] == 0:
            continue
        for b in range(n):
            if B[v, b] != 0 and B[a, b] != 0:
                return False
    return True


@njit(cache=False)
def _popcount(mask):
    count = 0
    while mask != 0:
        mask &= mask - np.uint64(1)
        count += 1
    return count


@njit(cache=False)
def _legal_data(B):
    """Return legal pairs and their 49-bit row masks for a valid graph.

    For an absent uv, the original test ``(B @ B @ B)[u, v] == 0`` says
    precisely that no edge crosses N(u) and N(v).  The bit-mask formulation
    performs that same test using one word intersection per possible middle
    vertex instead of a dense matrix product.
    """
    n = B.shape[0]
    neighbours = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        mask = np.uint64(0)
        for v in range(n):
            if B[u, v] != 0:
                mask |= np.uint64(1) << np.uint64(v)
        neighbours[u] = mask

    legal = np.zeros((n, n), dtype=np.bool_)
    legal_masks = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        for v in range(u + 1, n):
            if B[u, v] != 0:
                continue
            safe = True
            for a in range(n):
                if (neighbours[u] & (np.uint64(1) << np.uint64(a))) != 0:
                    if (neighbours[a] & neighbours[v]) != 0:
                        safe = False
                        break
            if safe:
                legal[u, v] = True
                legal[v, u] = True
                legal_masks[u] |= np.uint64(1) << np.uint64(v)
                legal_masks[v] |= np.uint64(1) << np.uint64(u)
    return legal, neighbours, legal_masks


@njit(cache=False)
def _loss_for_choices(neighbours, legal_masks, choices, iu, ju):
    """Return ``(B @ legal @ B)[u, v]`` for the candidate edge ids."""
    loss = np.empty(choices.size, dtype=np.int32)
    for index in range(choices.size):
        e = choices[index]
        u = iu[e]
        v = ju[e]
        total = 0
        for a in range(neighbours.size):
            if (neighbours[u] & (np.uint64(1) << np.uint64(a))) != 0:
                total += _popcount(legal_masks[a] & neighbours[v])
        loss[index] = total
    return loss


@njit(cache=False)
def _unlock_scores(B, present, iu, ju, edge_id):
    """Exact compiled counterpart of generation 38's ``unlock_scores``."""
    n = B.shape[0]
    m = iu.size
    credit = np.zeros(m, dtype=np.int32)

    for x in range(n):
        for y in range(x + 1, n):
            if B[x, y] != 0:
                continue

            count = 0
            a_path = -1
            b_path = -1
            for a in range(n):
                if B[x, a] == 0:
                    continue
                for b in range(n):
                    if B[a, b] != 0 and B[b, y] != 0:
                        count += 1
                        a_path = a
                        b_path = b
                        if count > 1:
                            break
                if count > 1:
                    break

            if count == 1:
                credit[edge_id[x, a_path]] += 1
                credit[edge_id[a_path, b_path]] += 1
                credit[edge_id[b_path, y]] += 1

    degree = np.zeros(n, dtype=np.int32)
    for u in range(n):
        for v in range(n):
            degree[u] += B[u, v]

    scores = np.empty(present.size, dtype=np.int32)
    for index in range(present.size):
        e = present[index]
        u = iu[e]
        v = ju[e]
        scores[index] = 18 * credit[e] + degree[u] + degree[v]
        if degree[u] > 7:
            scores[index] += degree[u] - 7
        if degree[v] > 7:
            scores[index] += degree[v] - 7
    return scores


@njit(cache=False)
def _first_bad_pair(B):
    """Return the first pair with two common neighbours, or (-1, -1)."""
    n = B.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in range(n):
                if B[u, w] != 0 and B[v, w] != 0:
                    common += 1
                    if common > 1:
                        return u, v
    return -1, -1


@lru_cache(maxsize=None)
def _problem_data(n):
    iu, ju = np.triu_indices(n, 1)
    iu = iu.astype(np.int32)
    ju = ju.astype(np.int32)
    m = iu.size
    edge_id = -np.ones((n, n), dtype=np.int32)
    edge_id[iu, ju] = np.arange(m, dtype=np.int32)
    edge_id[ju, iu] = np.arange(m, dtype=np.int32)
    return iu, ju, edge_id


def _saturate(B, iu, ju, source_mask, rng, banned=None, source_bias=0.30):
    """Generation 38 saturation with compiled legality and loss kernels."""
    B = B.copy()
    m = iu.size
    forbidden = np.zeros(m, dtype=bool) if banned is None else banned.copy()

    for _ in range(m):
        legal, neighbours, legal_masks = _legal_data(B)
        choices = np.flatnonzero(legal[iu, ju] & ~forbidden)
        if choices.size == 0 and np.any(forbidden):
            forbidden[:] = False
            choices = np.flatnonzero(legal[iu, ju])
        if choices.size == 0:
            break

        loss = _loss_for_choices(neighbours, legal_masks, choices, iu, ju)
        degree = B.sum(axis=1).astype(np.int32)
        cu = iu[choices]
        cv = ju[choices]
        load = degree[cu] + degree[cv]
        excess = 2 * np.maximum(degree[cu] - 7, 0)
        excess += 2 * np.maximum(degree[cv] - 7, 0)
        imbalance = np.abs(degree[cu] - degree[cv])
        score = 4 * loss + load + excess + imbalance // 2
        low = int(score.min())
        band = choices[score <= low + 1]
        if band.size == 0:
            band = choices[score == low]

        preferred = band[source_mask[band]]
        if preferred.size and rng.random() < source_bias:
            band = preferred

        e = int(band[int(rng.integers(band.size))])
        # e is selected from ``legal``.  The original defensive legal_edge
        # check is therefore guaranteed true and deliberately removed.
        u, v = int(iu[e]), int(ju[e])
        B[u, v] = 1
        B[v, u] = 1

    return B


def _source_order(style, source_edges, source_degree, iu, ju, rng):
    if source_edges.size == 0:
        return source_edges
    e = source_edges
    du, dv = source_degree[iu[e]], source_degree[ju[e]]
    load = du + dv
    product = du * dv
    imbalance = np.abs(du - dv)
    noise = rng.random(e.size)
    if style == 0:
        return e[np.lexsort((noise, load))]
    if style == 1:
        return e[np.lexsort((noise, -load))]
    if style == 2:
        return e[np.lexsort((noise, product))]
    if style == 3:
        return e[np.lexsort((noise, -product))]
    if style == 4:
        return e[np.lexsort((noise, imbalance))]
    if style == 5:
        return e[np.lexsort((noise, -imbalance))]
    if style == 6:
        central = np.abs(load - np.median(load))
        return e[np.lexsort((noise, central))]
    return e[rng.permutation(e.size)]


def _scaffold(style, limit, source_edges, source_degree, iu, ju, n, rng):
    B = np.zeros((n, n), dtype=np.uint8)
    order = _source_order(style, source_edges, source_degree, iu, ju, rng)
    if limit is not None:
        order = order[:min(int(limit), order.size)]
    for e in order:
        u, v = int(iu[e]), int(ju[e])
        if _is_legal_edge(B, u, v):
            B[u, v] = 1
            B[v, u] = 1
    return B


def _insert_archive(archive, B, iu, ju):
    signature = B[iu, ju].tobytes()
    for _, _, old_signature in archive:
        if signature == old_signature:
            return archive
    archive.append((int(B.sum() // 2), B.copy(), signature))
    archive.sort(key=lambda item: item[0], reverse=True)
    return archive[:ARCHIVE_SIZE]


def _repair(B):
    """Retain the original conservative final invariant-repair behaviour."""
    best = B.copy()
    n = best.shape[0]
    for _ in range(n * n):
        u, v = _first_bad_pair(best)
        if u < 0:
            break
        shared = np.flatnonzero(best[u] & best[v])
        if shared.size == 0:
            break
        degree = best.sum(axis=1)
        x = int(shared[np.argmax(degree[shared])])
        if degree[u] >= degree[v]:
            best[u, x] = 0
            best[x, u] = 0
        else:
            best[v, x] = 0
            best[x, v] = 0
    return best


def construct_new_graph(A, rng=None):
    """Return the generation-38 C4-free graph, accelerated for Axplorer."""
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square adjacency matrix")
    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    if n > 64:
        raise ValueError("turan_numba uses one 64-bit adjacency mask per vertex; N must be <= 64")
    iu, ju, edge_id = _problem_data(n)
    m = iu.size
    source = ((A != 0) | (A.T != 0)).astype(np.uint8)
    np.fill_diagonal(source, 0)
    source_mask = source[iu, ju].astype(bool)
    source_degree = source.sum(axis=1).astype(np.int32)
    source_edges = np.flatnonzero(source_mask)

    archive = []
    for plan_index, (style, limit) in enumerate(PLANS):
        state = _scaffold(style, limit, source_edges, source_degree, iu, ju, n, rng)
        state = _saturate(
            state, iu, ju, source_mask, rng,
            source_bias=0.46 if plan_index < 7 else 0.24,
        )
        archive = _insert_archive(archive, state, iu, ju)

    if not archive:
        state = _saturate(np.zeros((n, n), dtype=np.uint8), iu, ju, source_mask, rng, source_bias=0.0)
        archive = _insert_archive(archive, state, iu, ju)

    for trial, (remove_count, percentile) in enumerate(SCHEDULE):
        if len(archive) == 1 or rng.random() < 0.60:
            base = archive[0][1]
        else:
            base = archive[int(rng.integers(min(len(archive), 7)))][1]
        current = base.copy()
        present = np.flatnonzero(current[iu, ju])
        if present.size < remove_count:
            continue

        scores = _unlock_scores(current, present, iu, ju, edge_id)
        cutoff = np.percentile(scores, percentile)
        pool = present[scores >= cutoff]
        if pool.size < remove_count:
            pool = present

        selected = []
        used = set()
        for pick in range(remove_count):
            available = pool[~np.isin(pool, selected)]
            if available.size == 0:
                available = present[~np.isin(present, selected)]
            if pick and trial % 5 not in (0, 1):
                disjoint = np.array(
                    [e for e in available if int(iu[e]) not in used and int(ju[e]) not in used],
                    dtype=np.int64,
                )
                if disjoint.size:
                    available = disjoint
            e = int(available[int(rng.integers(available.size))])
            selected.append(e)
            used.add(int(iu[e]))
            used.add(int(ju[e]))

        banned = np.zeros(m, dtype=bool)
        for e in selected:
            u, v = int(iu[e]), int(ju[e])
            current[u, v] = 0
            current[v, u] = 0
            banned[e] = True

        # Removing edges cannot create a C4, and saturation only inserts
        # legal edges, so both original validity guards are redundant.
        candidate = _saturate(
            current, iu, ju, source_mask, rng, banned=banned,
            source_bias=0.18 if remove_count >= 5 else 0.32,
        )
        archive = _insert_archive(archive, candidate, iu, ju)

    best = _repair(archive[0][1])
    if _is_valid(best):
        best = _saturate(best, iu, ju, source_mask, rng, source_bias=0.20)
    if not _is_valid(best):
        for _ in range(n * n):
            u, v = _first_bad_pair(best)
            if u < 0:
                break
            shared = np.flatnonzero(best[u] & best[v])
            if shared.size == 0:
                break
            x = int(shared[0])
            best[u, x] = 0
            best[x, u] = 0

    np.fill_diagonal(best, 0)
    return best.astype(np.uint8)


def run_graph_construction(A, rng=None):
    """Alias intended for ``SquareDataPoint.local_search`` integration."""
    return construct_new_graph(A, rng=rng)
