"""Block until the NEXT checkpoint-eval result lands (one not yet reported), print a
one-line comparison vs the base model at the same g4096/c256 setting, record it as
reported, and exit -- so the caller gets a notification per dataset. Prints ALL_DONE
when every (tag,dataset) has been reported. Bounded so it never hangs forever."""
import json
import os
import sys
import time

TAGS = ["r1cot_ck2000", "r1cot_ck3000"]
DATASETS = ["math500", "minerva", "olympiad", "amc", "aime24", "aime25"]
MARKER = "eval_out/.reported_ckpt_evals"
MAX_WAIT_S = 6 * 3600


def score(path):
    d = json.load(open(path))
    return (d["score"] if "score" in d else d["mean"]) * 100.0, d.get("n")


def base_file(ds):
    return f"eval_out/mathbench_{ds}_g4096_c256.json"


def ckpt_file(tag, ds):
    return f"eval_out/mathbench_{tag}_{ds}_g4096_c256.json"


def reported():
    return set(open(MARKER).read().split()) if os.path.exists(MARKER) else set()


t0 = time.time()
while time.time() - t0 < MAX_WAIT_S:
    done = reported()
    for tag in TAGS:
        for ds in DATASETS:
            key = f"{tag}/{ds}"
            cf = ckpt_file(tag, ds)
            if key in done or not os.path.exists(cf):
                continue
            try:
                cs, n = score(cf)
            except Exception:
                continue  # file mid-write
            bf = base_file(ds)
            if os.path.exists(bf):
                bs, _ = score(bf)
                print(f"RESULT {tag} {ds}: {cs:.1f}%  |  base {bs:.1f}%  |  delta {cs - bs:+.1f}  (n={n})")
            else:
                print(f"RESULT {tag} {ds}: {cs:.1f}%  |  base (missing)  (n={n})")
            with open(MARKER, "a") as f:
                f.write(key + "\n")
            sys.exit(0)
    if len(done) >= len(TAGS) * len(DATASETS):
        print("ALL_DONE")
        sys.exit(0)
    time.sleep(60)
print("TIMEOUT")
