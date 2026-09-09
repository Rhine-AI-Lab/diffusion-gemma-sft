"""Build the Countdown (cd3) RL training jsonl from Jiayi-Pan/Countdown-Tasks-3to4.

Matches the d1/GDSD setup: full train split restricted to 3 numbers, target/nums pairs
(no supervised solution traces -- the reward is a verifier). Writes:
  dataset/countdown_cd3_train_full.jsonl   full cd3 pool (matches d1/GDSD exactly)
  dataset/countdown_cd3_train_dedup.jsonl  full pool MINUS the 256-puzzle test set

The 256-puzzle eval set (dataset/countdown_cd3_test.jsonl, d1's synthetic cd3 test)
ships with the repo. Note: the full pool overlaps ~29% with that test set, which d1/GDSD
do not deduplicate; use the *dedup* file for a leakage-free held-out number.

    python dataset/build_countdown_data.py
"""
import json
import os
import random

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").filter(lambda x: len(x["nums"]) == 3)
rows = [{"nums": list(r["nums"]), "target": int(r["target"])} for r in ds]
random.seed(0)
random.shuffle(rows)
with open(f"{HERE}/countdown_cd3_train_full.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

test = [json.loads(l) for l in open(f"{HERE}/countdown_cd3_test.jsonl")]
kte = {(tuple(sorted(int(x) for x in str(r["input"]).split(","))), int(r["output"])) for r in test}
dedup = [r for r in rows if (tuple(sorted(r["nums"])), r["target"]) not in kte]
with open(f"{HERE}/countdown_cd3_train_dedup.jsonl", "w") as f:
    for r in dedup:
        f.write(json.dumps(r) + "\n")

print(f"full={len(rows)}  dedup={len(dedup)}  (removed {len(rows) - len(dedup)} test-overlap rows)")
