# EVOLVE-BLOCK-START
"""Greedy algorithm for constructing square-free graphs with n vertices and many edges"""

import numpy as np

def construct_new_graph(A, rng = None):
    """First removes all 4-cycles in the given graph with adjacency matrix A,
    then greedily adds edges in such a way that no 4-cycles are introduced."""

    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if rng is None:
        rng = np.random.default_rng()

    n = A.shape[0]
    C = A @ A       #C[i,j] is the number of neighbours common to both vertex i and j

    bad_i, bad_j = np.where(np.triu(C > 1, k = 1))

    for i, j in zip(bad_i, bad_j):
        common = np.flatnonzero(A[i] & A[j])

        if len(common) > 1:
            keep_index = rng.integers(len(common))
            keep = common[keep_index]
            other = common[common != keep]

            choices = rng.integers(0, 2, size = len(other))

            remove_from_row_i = other[choices == 0]
            remove_from_row_j = other[choices == 1]

            A[i, remove_from_row_i] = 0
            A[remove_from_row_i, i] = 0

            A[j, remove_from_row_j] = 0
            A[remove_from_row_j, j] = 0
    
    # We have removed all 4-cycles, and now greedily add edges without creating 4-cycles

    for i in range(n-1):
        for j in range(i+1, n):
            if A[i][j] == 0:
                neighbours_i = np.flatnonzero(A[i])
                neighbours_j = np.flatnonzero(A[j])

                # adding an edge from i to j creates a c4 iff there already exists an edge between a neighbour of i and a neighbour of j
                creates_c4 = np.any(A[np.ix_(neighbours_i, neighbours_j)])      # submatrix with rows given by indices in neighbours_i and columns given by indices in neighbours_j

                if not creates_c4:
                    A[i][j] = 1
                    A[j][i] = 1
    
    return A 

# EVOLVE-BLOCK-END


# The following code remains fixed (not evolved)

def run_graph_construction(A, rng = None):
    """Run the graph construction algorithm on A"""
    return construct_new_graph(A = A, rng = rng)
