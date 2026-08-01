#!/usr/bin/env python3
import argparse

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

search_task_sys_msg = """You are an expert mathematician and algorithm designer working on the Turan problem for C4-free graphs.

Goal: develop a heuristic which, when applied to arbitrary 49-by-49 adjacency matrix A, yields a graph with no 4-cycles and as many edges as possible. The algorithm should directly improve A by
adding and deleting edges, and should not simply return a fixed matrix which is known to give a good score.

Key constraints:
- The evaluator scores mean improvement over a frozen seed algorithm on the same input matrices and rng seeds, so optimize for consistent per-input improvement rather than just the absolute edge count of one returned graph.
- Your algorithm should consist of adding and deleting edges to and from the input matrix - do not hard code a graph construction that ignores A, even if it gives a good score.
- Do not hard-code adjacency matrix or known solutions; write a general algorithm.
- Preserve the function signature construct_new_graph(A, rng=None). You may use rng for randomized choices, or ignore it for deterministic algorithms.
- Preserve the return type of construct_new_graph(A, rng=None): a numpy uint8 adjacency matrix.
- Do not change code outside the evolve block.
- Do not introduce dependencies beyond numpy unless clearly necessary.
- The evaluator runs this function on 100 different n=49 matrices. Keep runtime comfortably under 1 second per matrix. Avoid unbounded while loops, large numbers of restarts, repeated full matrix-cube computations inside long loops, or expensive local search unless iteration counts are small and fixed.
- Validity is mandatory. A graph with many edges but any 4-cycle receives no useful credit. After every construction or local-search phase, explicitly verify or preserve the invariant that no pair of vertices has more than one common neighbor. Be conservative: prefer a slightly lower edge count that always validates over an aggressive construction that may create C4s.
- If making a large structural change, prefer rewriting the full evolve block rather than using a fragile search/replace edit. Preserve the fixed wrapper below the evolve block and preserve the function name/signature construct_new_graph(A, rng=None).

Key directions to explore: 
- Use the characterization that a graph is C4-free iff every pair of vertices has at most one common neighbor. Equivalently, if A is the adjacency matrix, then the graph is C4-free iff A @ A has no off-diagonal entries greater than 1.
- Try greedy edge addition in randomized or strategically ordered edge lists. The order of attempted edges can matter a lot.
- Balance vertex degrees. Dense C4-free extremal graphs are often close to regular or nearly regular, so avoid creating very high-degree vertices too early unless justified.
- Use local search: remove a small number of edges to unlock the addition of more edges later. A locally maximal graph under single-edge additions may still be far from best.
- Try edge swaps: delete one or two low-value edges, then greedily refill without creating C4s.
- Use randomized restarts controlled by rng. Deterministic algorithms are allowed, but randomized repair and augmentation may find better graphs.
- Use the fact you are only considering graphs with 49 vertices. This may help in finding certain symmetries, e.g. by considering certain groupings of vertices. 
- Use the input graph A as the actual search state. You may remove and add edges, but do not discard A wholesale and replace it with a pre-existing construction.
- After repairing all C4 violations, perform a saturation pass: repeatedly add every edge that can be added without creating a C4 until no such edge remains.
- Track candidate edge additions by how many future additions they block, preferring edges that preserve many remaining possibilities.
"""


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="00:05:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_small.yaml")
    args = parser.parse_args()
    main(args.config_path)
