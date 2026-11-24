"""PREREG_main.md 2026-11-24 (9): C0 / VIS / ORACLE on H3 (T1@hid) eval 0-39. Paired McNemar exact + Wald CI."""
import json, glob, math, os, sys, collections
D = sys.argv[1] if len(sys.argv) > 1 else "main/runs"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "eval"
R = {}
for c, tag in (("C0", "C0_side9"), ("VIS", "VIS_side9"), ("ORACLE", "ORACLE_side9")):
    for f in glob.glob(f"{D}/{tag}/T1_hid_{SPLIT}*.json"):
        d = json.load(open(f)); k = os.path.basename(f)[:-5]
        h = d["collision"]["hidden"]; s = bool(d["success"]); col = bool(d["collision"]["safelibero_1mm"])
        ev = h["events"]
        cause = "none"
        if ev:
            e = ev[0]
            if "robot" in e["touch"]: cause = "robot"
            elif e["held"] and "held" in e["touch"]: cause = "held"
            elif "other_object" in e["touch"] or "held" in e["touch"]: cause = "released"
            else: cause = "shake"
        R.setdefault(c, {})[k] = dict(s=s, safe=s and not col, col=col, box=h["box_moved_1mm"], cause=cause,
                                      steps=d["steps"], seen=h["first_seen_step"], move=h["first_move_step"])
keys = sorted(set.intersection(*[set(v) for v in R.values()]))
N = len(keys)
def mc(a, b, f):
    n10 = sum(R[a][k][f] and not R[b][k][f] for k in keys); n01 = sum(R[b][k][f] and not R[a][k][f] for k in keys)
    n = n10 + n01
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    dl = (n10 - n01) / N; se = math.sqrt(max(((n10 + n01) / N - dl ** 2) / N, 0))
    return n10, n01, p, 100 * (dl - 1.96 * se), 100 * (dl + 1.96 * se)
print("paired n =", N)
for c in R:
    rr = [R[c][k] for k in keys]
    succ = [r["steps"] for r in rr if r["s"]]
    print(f"{c:7s} TSR {sum(r['s'] for r in rr):2d}  SSR {sum(r['safe'] for r in rr):2d}  CAR {sum(not r['col'] for r in rr):2d}"
          f"  box_moved {sum(r['box'] for r in rr):2d}  ETS(succ) {sum(succ)/max(len(succ),1):.0f}  causes {dict(collections.Counter(r['cause'] for r in rr if r['box']))}")
for a, b in (("VIS", "C0"), ("ORACLE", "VIS"), ("ORACLE", "C0")):
    for f, nm in (("safe", "SSR"), ("s", "TSR")):
        n10, n01, p, lo, hi = mc(a, b, f)
        print(f"{a} vs {b} {nm}: only {a} {n10}, only {b} {n01}, p={p:.3g}, 95% CI [{lo:+.1f}, {hi:+.1f}] pt")
