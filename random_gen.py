import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random

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

def random_adjacency_matrix(n, p = 0.5, seed = None):   
    """Generates a random graph with n vertices by constructing its adjacency matrix."""
    rng = np.random.default_rng(seed)
    upper = rng.random((n,n)) < p
    upper = np.triu(upper, k = 1)
    A = upper + upper.T 

    return A.astype(int)


def cycle_checker(A):
    """Returns True if the graph with adjacency matrix A contains a 4-cycle, and False otherwise."""
    n = A.shape[0]

    for i in range(n-1):
        for j in range(i+1, n):
            if np.dot(A[i], A[j]) > 1:      # equivalently the ij entry of A^2 is > 1
                return True
            
    return False 


def cycle_remover(A, seed = None):
    """Removes all 4-cycles in a given graph with adjacency matrix A. For each fixed pair of rows, 
    we find the columns where the entries in both rows are 1. If there is more than one such column, 
    we choose one random column in which to keep both entries as 1, and for each of other columns, we 
    choose one of the two rows at random and change its entry in that column to 0."""
    n = A.shape[0]

    rng = np.random.default_rng(seed)
    for i in range(n-1):
        for j in range(i+1, n):
            matching_ones = []

            for k in range(n):
                if A[i][k] == 1 and A[j][k] == 1:
                    matching_ones.append(k)

            if len(matching_ones) > 1:
                keep = random.choice(matching_ones)
                matching_ones.remove(keep)

                for l in matching_ones:
                    num = rng.random()

                    if num < 0.5:
                        A[i][l] = 0
                        A[l][i] = 0
                        
                    else:
                        A[j][l] = 0
                        A[l][j] = 0

    return A 


def edge_adder(A):
    """Greedy algorithm for adding edges to a graph with adjacency matrix A in such a way that no
     4-cycles are introduced. Input must have no 4-cycles. For each zero entry in the graph, we check
     whether changing it to a 1 preserves the property of being free of a 4-cycle. If so, we make
     this change."""
    if cycle_checker(A):
        raise ValueError('Input matrix must have no 4-cycles')
    
    n = A.shape[0]
    
    for i in range(n-1):
        for j in range(i+1, n):
            if A[i][j] == 0:
                A[i][j] = 1
                A[j][i] = 1

                if cycle_checker(A):
                    A[i][j] = 0 
                    A[j][i] = 0

    return A


def greedy_vs_max(min_n = 5, max_n = 30, iters = 10**2):
    greedy_vs_max = defaultdict(list)
    for n in range(min_n, max_n + 1):
        max_found = 0
        for _ in range(iters):
            A = random_adjacency_matrix(n)
            A = cycle_remover(A)
            A = edge_adder(A)
            max_found = int(max(max_found, A.sum()/2))
        greedy_vs_max[n] = [max_found, ex_C4[n]]

    x = sorted(greedy_vs_max.keys())
    greedy_values = [greedy_vs_max[n][0] for n in x]
    max_values = [greedy_vs_max[n][1] for n in x]

    plt.figure(figsize = (12,6))
    plt.plot(x, greedy_values, marker = 'o', label = 'Greedy')
    plt.plot(x, max_values, marker = 'o', label = 'Maximum')
    plt.xlabel('Number of vertices')
    plt.ylabel('Number of edges')
    plt.title(f'Edge count for square-free graphs with $n$ vertices: greedy algorithm ({iters} iterations) vs maximum possible')
    plt.legend()
    plt.grid(True)

    plt.show()



print(greedy_vs_max())


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

# plotter()    

