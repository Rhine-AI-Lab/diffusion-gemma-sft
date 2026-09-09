import json, os, time
BASE="eval_out/mathbench_{tag}{ds}_g4096_c256.json"
DS=["math500","minerva","olympiad","amc","aime24","aime25"]
CKPTS=["r1cot_ck1000","r1cot_ck2000","r1cot_ck3000"]
def sc(p):
    try:
        d=json.load(open(p)); return (d.get("score", d.get("mean"))*100)
    except: return None
# wait until ck2000+ck3000 (the currently-running ones) are all present
need=[BASE.format(tag=c+"_", ds=d) for c in ["r1cot_ck2000","r1cot_ck3000"] for d in DS]
t0=time.time()
while time.time()-t0 < 6*3600:
    if all(os.path.exists(f) for f in need): break
    time.sleep(60)
base={d: sc(BASE.format(tag="", ds=d)) for d in DS}
print("\n=== r1cot checkpoint trajectory @ g4096/c256 (%) ===")
print(f"{'dataset':11s} {'base':>6s} {'ck1000':>7s} {'ck2000':>7s} {'ck3000':>7s}")
for d in DS:
    row=f"{d:11s} {base[d]:6.1f}"
    for c in CKPTS:
        v=sc(BASE.format(tag=c+"_", ds=d))
        row += f" {('%.1f'%v) if v is not None else '  -  ':>7s}"
    print(row)
print("[collect] DONE")
