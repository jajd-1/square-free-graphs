"""
Evaluator for square-free graph construction
"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import frozen_seed_algorithm

from shinka.core import run_shinka_eval

script_directory = Path(__file__).resolve().parent
npz_path = script_directory/'starting_matrices'/'n49_evaluation.npz'

with np.load(npz_path) as data:
    matrices = data['matrices']
    matrix_seeds = data['matrix_seeds']
    matrix_densities = data['matrix_densities']


def validate_solution(A) -> Tuple[bool, str]: 
    """Check if adjacency matrix with no 49-cycles"""

    msg = 'Matrix is an adjacency matrix of a graph with no 4-cycles.'

    if A.dtype != np.uint8:
        msg = 'A must have dtype np.uint8'
        return False, msg

    if A.shape[0] != A.shape[1]:
        msg = 'Matrix not square.'
        return False, msg
    
    if A.shape != matrices[0].shape:
        msg = f"Matrix has shape {A.shape}, expected {matrices[0].shape}"
        return False, msg

    for i in range(A.shape[0]):
        if A[i][i] != 0:
            msg = 'Non-zero diagonal entry.'
            return False, msg
        
        for j in range(i+1, A.shape[0]):
            if A[i][j] not in [0,1]:
                msg = 'Off-diagonal entry not 0 or 1.'
                return False, msg
            if A[i][j] != A[j][i]:
                msg = 'Matrix not symmetric.'
                return False, msg

    C = A @ A 
    np.fill_diagonal(C, 0)

    if np.any(C>1):
        msg = 'Graph is not square-free.'
        return False, msg 
    
    return True, msg 

EVAL_RNG_BASE_SEED = 12345
new_graph_construction_rng_seeds = np.arange(EVAL_RNG_BASE_SEED, EVAL_RNG_BASE_SEED + len(matrices))

seed_algorithm_results = [frozen_seed_algorithm.run_graph_construction(matrices[run_index].copy(), np.random.default_rng(int(new_graph_construction_rng_seeds[run_index]))) for run_index in range(len(matrices))]

seed_algorithm_edge_counts = [int(A.sum() / 2) for A in seed_algorithm_results]


def get_kwargs(run_index) -> Dict:
    """Provides keyword arguments for run_graph_construction"""
    return {
        "A": matrices[run_index].copy(),
        "rng": np.random.default_rng(int(new_graph_construction_rng_seeds[run_index])),
    }

def aggregate_graph_construction_metrics(results, results_dir) -> Dict:
    """Aggregates metrics for 4-cycle-free graph construction, e.g. total number of edges, 
    distance to best known bound, length of code"""

    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}
    
    edge_counts = [int(A.sum()/2) for A in results]
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

    #combined_score is the one Shinka tries to improve

def main(program_path: str, results_dir: str):
    """Runs the graph construction using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_experiment_runs = len(matrices)

    # Define a nested function to pass results_dir to the aggregator
    def _aggregator_with_context(results) -> Dict:
        return aggregate_graph_construction_metrics(results, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_graph_construction",
        num_runs=num_experiment_runs,
        get_experiment_kwargs=get_kwargs,
        validate_fn=validate_solution,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Square-free graph construction evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_graph_construction')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Dir to save results (metrics.json, correct.json, extra.npz)",
    )
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
