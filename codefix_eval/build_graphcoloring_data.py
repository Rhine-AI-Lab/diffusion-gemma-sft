"""Synthetic planted k-coloring SFT data with an exact-chromatic-number
certificate (no solver needed): plant k nonempty color classes, add a
k-clique across one representative per class (proves chi(G) >= k), add
remaining edges only between different classes (planted coloring proves
chi(G) <= k) -- so chi(G) = k exactly, by construction, at zero solve cost.

Output is a fixed JSON dict {vertex_id: color_id}, closed/low-cardinality
per vertex (only k choices) and structurally simple like sudoku's grid,
rather than CRUXEval's arbitrary values -- the shape this session's
output-entropy theory predicts should be an SFT win.
"""

from __future__ import annotations

import argparse
import json
import random

_PROMPT_TEMPLATE = """You are given an undirected graph with {n} vertices, labeled 0 to {n_minus_1}.
It is known that this graph can be properly colored using at most {k} colors (single digits 0 to {k_minus_1}), meaning no edge connects two vertices of the same color.

Edges:
{edges}

Output a valid {k}-coloring for these {n} vertices as a single string of EXACTLY {n} digits (no separators, no other text), where the digit at position i (0-indexed) is vertex i's color. This graph has {n} vertices, so your answer must be exactly {n} characters long. Example for a 5-vertex, 3-color graph: 02110"""


def _plant_instance(n: int, k: int, density: float, rng: random.Random, planted_clique: bool = True):
    """Returns (edges, coloring) with edges a sorted list of [u, v] pairs
    (u < v after the final vertex permutation) and coloring a dict
    vertex_id -> color_id (0..k-1), canonicalized by first appearance."""
    # Assign one vertex per class first (guarantees nonempty classes and
    # gives us clique representatives), then distribute the rest uniformly.
    order = list(range(n))
    rng.shuffle(order)
    class_of = {}
    reps = order[:k]
    for c, v in enumerate(reps):
        class_of[v] = c
    for v in order[k:]:
        class_of[v] = rng.randrange(k)

    edges = set()
    if planted_clique:
        for i in range(k):
            for j in range(i + 1, k):
                u, v = reps[i], reps[j]
                edges.add((min(u, v), max(u, v)))

    for i in range(n):
        for j in range(i + 1, n):
            if class_of[i] == class_of[j]:
                continue
            if (i, j) in edges:
                continue
            if rng.random() < density:
                edges.add((i, j))

    # Permute vertex IDs so the planted structure isn't recoverable by
    # position (e.g. clique reps or class order) alone.
    perm = list(range(n))
    rng.shuffle(perm)
    edges = sorted((min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in edges)
    coloring = {perm[v]: c for v, c in class_of.items()}

    # Canonicalize color LABELS by order of first appearance (vertex id
    # order) so the training target is deterministic, without touching
    # which vertices share a class.
    relabel = {}
    canon = {}
    for v in range(n):
        c = coloring[v]
        if c not in relabel:
            relabel[c] = len(relabel)
        canon[v] = relabel[c]
    return edges, canon


def parse_coloring(digits: str, n: int) -> dict | None:
    digits = digits.strip()
    if len(digits) != n or not digits.isdigit():
        return None
    return {i: int(digits[i]) for i in range(n)}


def is_valid_coloring(edges, coloring: dict, k: int, n: int) -> tuple[bool, str]:
    if set(coloring.keys()) != set(range(n)):
        return False, "coloring doesn't cover exactly vertices 0..n-1"
    if any(not (0 <= c < k) for c in coloring.values()):
        return False, f"color id outside [0, {k})"
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False, f"edge ({u},{v}) monochromatic"
    return True, "ok"


def make_example(n: int, k: int, density: float, rng: random.Random, planted_clique: bool = True) -> dict:
    edges, coloring = _plant_instance(n, k, density, rng, planted_clique=planted_clique)
    ok, msg = is_valid_coloring(edges, coloring, k, n)
    assert ok, f"generator produced invalid instance: {msg}"
    edge_str = ",".join(f"{u}-{v}" for u, v in edges)
    prompt = _PROMPT_TEMPLATE.format(n=n, n_minus_1=n - 1, k=k, k_minus_1=k - 1, edges=edge_str)
    completion = "".join(str(coloring[v]) for v in range(n))
    return {
        "prompt": prompt,
        "completion": completion,
        "n": n,
        "k": k,
        "num_edges": len(edges),
        "density": density,
    }


_ID_N = [16, 24, 32, 40, 48, 64]
_ID_K = [3, 4, 5, 6]


def _gen_split(count: int, rng: random.Random, n_choices, k_choices, density_range, planted_clique=True):
    out = []
    for _ in range(count):
        n = rng.choice(n_choices)
        k = rng.choice(k_choices)
        density = rng.uniform(*density_range)
        out.append(make_example(n, k, density, rng, planted_clique=planted_clique))
    return out


def _write(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} examples -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="codefix_eval/graphcoloring")
    ap.add_argument("--n_train", type=int, default=500_000)
    ap.add_argument("--n_val", type=int, default=5_000)
    ap.add_argument("--n_test_id", type=int, default=10_000)
    ap.add_argument("--n_test_size", type=int, default=5_000)
    ap.add_argument("--n_test_ood", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import os

    os.makedirs(args.out_dir, exist_ok=True)
    # Kept low: edge count (and prompt length) scales with density * n^2 *
    # (k-1)/k, and even a modest density blows up token counts fast for
    # n up to 64 (or 128 on test_size) -- see SFT_TASK_SELECTION_NOTES.md
    # for the measured token-length blowup at density=0.3-0.7.
    density_range = (0.06, 0.18)

    rng = random.Random(args.seed)
    _write(_gen_split(args.n_train, rng, _ID_N, _ID_K, density_range), f"{args.out_dir}/train.jsonl")

    rng = random.Random(args.seed + 1)
    _write(_gen_split(args.n_val, rng, _ID_N, _ID_K, density_range), f"{args.out_dir}/validation.jsonl")

    rng = random.Random(args.seed + 2)
    _write(_gen_split(args.n_test_id, rng, _ID_N, _ID_K, density_range), f"{args.out_dir}/test_id.jsonl")

    rng = random.Random(args.seed + 3)
    _write(
        _gen_split(args.n_test_size, rng, [72, 88, 104, 128], _ID_K, density_range),
        f"{args.out_dir}/test_size.jsonl",
    )

    # OOD: density shifted away from train's 0.06-0.18 band (sparser or
    # somewhat denser, but still capped to keep prompts tractable -- going
    # to 0.85+ would blow up edge count to O(n^2) and the token budget with
    # it), and no planted clique, so the optimality certificate (and
    # therefore the difficulty profile) differs from every other split.
    rng = random.Random(args.seed + 4)
    ood_rows = []
    for _ in range(args.n_test_ood):
        n = rng.choice(_ID_N)
        k = rng.choice(_ID_K)
        density = rng.choice([rng.uniform(0.01, 0.04), rng.uniform(0.22, 0.32)])
        ood_rows.append(make_example(n, k, density, rng, planted_clique=False))
    _write(ood_rows, f"{args.out_dir}/test_ood.jsonl")


if __name__ == "__main__":
    main()
