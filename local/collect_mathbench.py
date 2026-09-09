#!/usr/bin/env python
"""Aggregate the math-benchmark sweep result JSONs into one table.

Reads eval_out/mathbench_<dataset>_g<gen>_c<canvas>.json (written by
run_mathbench.sh) and prints a datasets x (gen, #canvases) accuracy table.
Non-AIME cells are pass@1 accuracy; AIME cells are avg@k mean +/- std.

    python local/collect_mathbench.py [--glob 'eval_out/mathbench_*.json']
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

TAG_RE = re.compile(r"mathbench_(?P<ds>[a-z0-9]+)_g(?P<gen>\d+)_c(?P<canvas>\d+)\.json$")
DATASET_ORDER = ["math500", "minerva", "olympiad", "amc", "aime24", "aime25"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="eval_out/mathbench_*.json")
    ap.add_argument("--md", default=None, help="Also append a markdown table to this file.")
    args = ap.parse_args()

    # cell[(dataset, gen, canvas)] = formatted accuracy string
    cells: dict[tuple, str] = {}
    cols: set[tuple[int, int]] = set()          # (gen, canvas)
    datasets_seen: list[str] = []
    for path in sorted(glob.glob(args.glob)):
        m = TAG_RE.search(os.path.basename(path))
        if not m:  # skip shard files / anything off-pattern
            continue
        ds, gen, canvas = m["ds"], int(m["gen"]), int(m["canvas"])
        try:
            d = json.load(open(path))
        except Exception as e:
            cells[(ds, gen, canvas)] = f"ERR({type(e).__name__})"
            cols.add((gen, canvas)); continue
        if "mean" in d:  # avg@k
            cells[(ds, gen, canvas)] = f"{d['mean'] * 100:.1f}±{d['std'] * 100:.1f}"
        elif "score" in d:
            cells[(ds, gen, canvas)] = f"{d['score'] * 100:.1f}"
        else:
            cells[(ds, gen, canvas)] = "?"
        cols.add((gen, canvas))
        if ds not in datasets_seen:
            datasets_seen.append(ds)

    if not cells:
        print(f"[collect] no result files matched {args.glob}")
        return

    ncanv = lambda gen, canvas: max(1, gen // canvas)
    ordered_cols = sorted(cols, key=lambda gc: (gc[0], ncanv(*gc)))
    header = ["dataset"] + [f"g{gen}/{ncanv(gen, canvas)}c" for gen, canvas in ordered_cols]
    rows_ds = [d for d in DATASET_ORDER if d in datasets_seen] + \
              [d for d in datasets_seen if d not in DATASET_ORDER]

    def fmt_row(ds):
        return [ds] + [cells.get((ds, gen, canvas), "-") for gen, canvas in ordered_cols]

    table = [header] + [fmt_row(ds) for ds in rows_ds]
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    sep = "  "
    lines = [sep.join(c.ljust(widths[i]) for i, c in enumerate(header))]
    lines.append(sep.join("-" * widths[i] for i in range(len(header))))
    lines += [sep.join(c.ljust(widths[i]) for i, c in enumerate(fmt_row(ds))) for ds in rows_ds]
    out = "\n".join(lines)
    print("\n=== math-benchmark sweep (accuracy %; AIME = mean±std over seeds) ===")
    print(out)

    if args.md:
        md = ["", "### Math-benchmark sweep (accuracy %; AIME = avg@k mean±std)", ""]
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join("---" for _ in header) + " |")
        for ds in rows_ds:
            md.append("| " + " | ".join(fmt_row(ds)) + " |")
        with open(args.md, "a") as fh:
            fh.write("\n".join(md) + "\n")
        print(f"\n[collect] appended markdown table to {args.md}")


if __name__ == "__main__":
    main()
