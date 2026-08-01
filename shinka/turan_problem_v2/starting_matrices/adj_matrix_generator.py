import numpy as np
from collections.abc import Iterable
from pathlib import Path

script_directory = Path(__file__).resolve().parent

def random_adjacency_matrix(n, p, rng = None):   
    """Generates a random graph with n vertices by constructing its adjacency matrix."""

    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((n,n), dtype = np.uint8)

    upper_indices = np.triu_indices(n, k = 1)
    A[upper_indices] = rng.random(len(upper_indices[0])) < p 
    A = A + A.T 

    return A


def create_evaluation_and_test_matrices(n: int, seeds: Iterable[int], output_path: Path):
    densities = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype = np.float32)
    seed_array = np.asarray(list(seeds), dtype = np.int32)

    matrices = np.stack([random_adjacency_matrix(n = n, p = float(p), rng = np.random.default_rng(np.random.SeedSequence([int(seed), density_index]))) 
                                                for density_index, p in enumerate(densities)
                                                for seed in seed_array])
    
    matrix_densities = np.repeat(densities, len(seed_array))
    matrix_seeds = np.tile(seed_array, len(densities))
    matrix_density_indices = np.repeat(np.arange(len(densities), dtype=np.int32), len(seed_array))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    np.savez_compressed(output_path, n = np.asarray(n), density_values = densities, matrix_seeds = matrix_seeds,
                        matrix_densities = matrix_densities, matrix_density_indices = matrix_density_indices, matrices = matrices)

    return matrices


create_evaluation_and_test_matrices(49, range(20), Path(script_directory/"n49_evaluation.npz"))
create_evaluation_and_test_matrices(49, range(20, 40), Path(script_directory/"n49_test.npz"))






