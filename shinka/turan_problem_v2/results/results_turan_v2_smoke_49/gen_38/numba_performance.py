import os
import numpy as np
from pathlib import Path
from typing import Dict
from turan_numba import run_graph_construction
import json 
import frozen_seed_algorithm
from time import perf_counter

start = perf_counter()

example_directory = Path(__file__).resolve().parent.parent.parent.parent
npz_path = example_directory/'starting_matrices'/'n49_evaluation.npz'


with np.load(npz_path) as data:
    matrices = data['matrices']
    matrix_seeds = data['matrix_seeds']
    matrix_densities = data['matrix_densities']

results_dir = Path(__file__).resolve().parent
results = matrices, matrix_seeds, matrix_densities

EVAL_RNG_BASE_SEED = 12345
new_graph_construction_rng_seeds = np.arange(EVAL_RNG_BASE_SEED, EVAL_RNG_BASE_SEED + len(matrices))

seed_algorithm_results = [frozen_seed_algorithm.run_graph_construction(matrices[run_index].copy(), np.random.default_rng(int(new_graph_construction_rng_seeds[run_index]))) for run_index in range(len(matrices))]

seed_algorithm_edge_counts = [int(A.sum() / 2) for A in seed_algorithm_results]


def aggregate_graph_construction_metrics(matrices, matrix_seeds, matrix_densities, results_dir) -> Dict:

    edge_counts = []

    for run_index, matrix in enumerate(matrices):
        for _ in range(5):
            new_matrix = run_graph_construction(matrix, np.random.default_rng(int(new_graph_construction_rng_seeds[run_index])))
        edge_counts.append(int(new_matrix.sum()/2))

    edge_improvements = [
        int(new - baseline)
        for new, baseline in zip(edge_counts, seed_algorithm_edge_counts)
    ]
    

    metrics = {
        "combined_score": float(np.mean(edge_improvements)),     #combined_score is the one Shinka tries to improve

        "public": {
            "mean_edges": float(np.mean(edge_counts)),
            "median_edges": int(np.median(edge_counts)),
            "max_edges": int(np.max(edge_counts)),
            "mean_improvement_over_seed": float(np.mean(edge_improvements)),
            "max_improvement_over_seed": int(np.max(edge_improvements)),
        }, 

        "private": {
            "num_graphs": len(results),
            "edge_count_list": edge_counts,
            "seed_algorithm_edge_count_list": seed_algorithm_edge_counts,
            "edge_improvement_over_seed_list": edge_improvements,
            "matrix_seed_list": matrix_seeds.tolist(),
            "improved_graph_seed_list": new_graph_construction_rng_seeds.tolist(),
            "initial_density_list": matrix_densities.tolist(),
        },
    }

    extra_file = os.path.join(results_dir, "extra.npz")     #save detailed graph data not contained in metrics

    try:
        np.savez(
            extra_file, 
            graphs = np.stack(results),
            edge_counts = np.array(edge_counts, dtype = np.int16),
            matrix_seeds = matrix_seeds, 
            matrix_densities = matrix_densities,
        )
    
    except Exception as e:
        print(f"Error saving extra.npz: {e}")
        metrics["extra_npz_save_error"] = str(e)
    
    return metrics

metrics = aggregate_graph_construction_metrics(
    matrices,
    matrix_seeds,
    matrix_densities,
    results_dir,
)

metrics_path = results_dir/'numba_eval_metrics.json'

with metrics_path.open('w', encoding = 'utf-8') as f:
    json.dump(metrics, f, indent = 4)

elapsed = (perf_counter() - start)/5

print(f'Metrics on test data saved to {metrics_path}.')
print(f'Average time taken per run (Numba): {elapsed:.3f} seconds')