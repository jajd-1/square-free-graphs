import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import time
from concurrent.futures import ProcessPoolExecutor
import os


"""For n < 41, a list of the maximum number of edges a square-free graph with n vertices can possess.
For n <= 41 < 50, a list of the known lower bounds on the maximum number of such edges."""

ex_C4 = {
    4: 4,
    5: 6,
    6: 7,
    7: 9,
    8: 11,
    9: 13,
    10: 16,
    11: 18,
    12: 21,
    13: 24,
    14: 27,
    15: 30,
    16: 33,
    17: 36,
    18: 39,
    19: 42,
    20: 46,
    21: 50,
    22: 52,
    23: 56,
    24: 59,
    25: 63,
    26: 67,
    27: 71,
    28: 76,
    29: 80,
    30: 85,
    31: 90,
    32: 92,
    33: 96,
    34: 102,
    35: 106,
    36: 110,
    37: 113,
    38: 117,
    39: 122,
    40: 127,
    41: ">=132",
    42: ">=137",
    43: ">=142",
    44: ">=148",
    45: ">=154",
    46: ">=157",
    47: ">=163",
    48: ">=168",
    49: ">=174",
}

rng = np.random.default_rng()

def random_adjacency_matrix(n, p = 0.5, rng = None):   
    """Generates a random graph with n vertices by constructing its adjacency matrix."""

    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((n,n), dtype = np.uint8)

    upper_indices = np.triu_indices(n, k = 1)
    A[upper_indices] = rng.random(len(upper_indices[0])) < p 
    A = A + A.T 

    return A



def cycle_checker(A):
    """Returns True if the graph with adjacency matrix A contains a 4-cycle, and False otherwise."""

    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    C = A @ A 
    np.fill_diagonal(C, 0)
    
    return np.any(C > 1) 


def cycle_remover(A, rng = None):
    """Removes all 4-cycles in a given graph with adjacency matrix A."""

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
    
    return A


def edge_adder(A):
    """Greedy algorithm for adding edges to a graph with adjacency matrix A in such a way that no 4-cycles are introduced. 
    Input must have no 4-cycles."""
    
    if A.dtype != np.uint8:
        raise TypeError("A must have dtype np.uint8")

    if cycle_checker(A):
        raise ValueError('Input matrix must have no 4-cycles')
    
    n = A.shape[0]

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


# ------------------------------------------------------------


def single_trial(min_n = 15, max_n = 30, rng = None):

    if rng is None:
        rng = np.random.default_rng()

    edge_count = defaultdict(int)
    for n in range(min_n, max_n + 1):
        A = random_adjacency_matrix(n)
        A = cycle_remover(A)
        A = edge_adder(A)
        edge_count[n] = int(A.sum()/2)
    
    return edge_count


def batch_trial(args):
    min_n, max_n, batch_size, seed = args

    rng = np.random.default_rng(seed)
    best = defaultdict(int)

    for _ in range(batch_size):
        result = single_trial(min_n, max_n, rng = rng)

        for n, edges in result.items():
            best[n] = max(best[n], edges)
    
    return dict(best)


def parallel_trials(min_n, max_n, iters, num_workers, base_seed):
    if num_workers is None:
        num_workers = os.cpu_count()

    base = np.random.SeedSequence(base_seed)
    child_seeds = base.spawn(num_workers)

    batch_sizes = [iters // num_workers] * num_workers 
    for k in range (iters % num_workers):
        batch_sizes[k] += 1

    args = [(min_n, max_n, batch_sizes[k], child_seeds[k]) for k in range(num_workers) if batch_sizes[k] > 0]

    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        batch_results = list(executor.map(batch_trial, args))

    global_best = defaultdict(int)

    for result in batch_results:
        for n, edges in result.items():
            global_best[n] = max(global_best[n], edges)
    
    return dict(global_best)

if __name__ == '__main__':
    for workers in [4, 6, 8, 10]:
        start = time.perf_counter()

        result = parallel_trials(
            min_n=30,
            max_n=40,
            iters=1000,
            num_workers=workers,
            base_seed=1
        )

        end = time.perf_counter()

        print(f"{workers} workers: {end - start:.2f} seconds")




# if __name__ == '__main__':
#     iters = [10, 100, 1000, 5000]
#     greedy_results = defaultdict(dict)

#     for iter in iters:
#         greedy_results[iter] = parallel_trials(min_n = 20, max_n = 40, iters = iter, num_workers = None, base_seed = 1)
    
#     comparison = defaultdict(int)

#     for n in greedy_results[iters[0]].keys():
#         comparison[n] = [greedy_results[iter][n] for iter in iters] +  [ex_C4[n], math.floor((n/4)*(1 + math.sqrt(4*n - 3)))]

#     print(comparison)
    
#     x = sorted(comparison.keys())

#     greedy_values = []
#     for i in range(len(iters)):
#         greedy_values.append([comparison[n][i] for n in x])
        
#     max_values = [comparison[n][-2] for n in x]
#     reiman_values = [comparison[n][-1] for n in x]

#     plt.figure(figsize = (12,6))
#     for i, iter in enumerate(iters):
#         plt.plot(x, greedy_values[i], marker = 'o', markersize = 4, label = f'Greedy with {iter} iterations')
#     plt.plot(x, max_values, marker = 'o', markersize = 4, label = 'Maximum')
#     plt.plot(x, reiman_values, marker = 'o', markersize = 4, label = 'Reiman bound' )
#     plt.xlabel('Number of vertices')
#     plt.ylabel('Number of edges')
#     plt.title(f'Edge count for square-free graphs with $n$ vertices')
#     plt.legend()
#     plt.grid(True)

#     plt.show()














"""

A = random_adjacency_matrix(5)
print(cycle_checker(A), int(A.sum()/2))
B = cycle_remover(A)
print(cycle_checker(B), int(B.sum()/2))
C = edge_adder(B)
print(cycle_checker(C), int(C.sum()/2))



def draw_graph_from_adjacency_matrix(A):
    G = nx.from_numpy_array(A)
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_label = True, node_size = 700, font_size = 12)
    plt.show()


def plotter(min_n = 4, max_n = 10, iters = 10**5):
    no_cycle_percentage = defaultdict(int)
    for n in range(min_n, max_n + 1):
        count = 0 
        for _ in range(iters):
            A = random_adjacency_matrix(n)
            cycle_present = cycle_checker(A)
            if not cycle_present:
                count += 1
        no_cycle_percentage[n] = np.log(count)
        
    
    plt.figure(figsize = (12, 6))

    x = sorted(no_cycle_percentage.keys())
    y = [no_cycle_percentage[k] for k in x]

    plt.plot(x, y, marker = 'o')
    plt.xlabel('Number of vertices in graph')
    plt.ylabel('Percentage without 4-cycles')
    plt.title('Percentage of randomly generated graphs with $n$ vertices not containing any 4-cycles')
    plt.grid(True)
    plt.show()

plotter()    

"""