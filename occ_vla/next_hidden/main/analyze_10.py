"""PREREG_main.md 2026-11-25 (10): OBJ / LOOK / OBJLOOK (main10) against C0 / VIS of (9) on H3 eval 0-39."""
import json, glob, math, os, collections
TAGS = {"C0": "C0_side9", "VIS": "VIS_side9", "OBJ": "OBJ_main10", "LOOK": "LOOK_main10", "OBJLOOK": "OBJLOOK_main10"}
R = {}
for c, tag in TAGS.items():
    for f in glob.glob(f"main/runs/{tag}/T1_hid_eval*.json"):
        d = json.load(open(f)); k = os.path.basename(f)[:-5]
        h = d["collision"]["hidden"]; s = bool(d["success"]); col = bool(d["collision"]["safelibero_1mm"])
        cause = "none"
        if h["events"]:
            e = h["events"][0]
            cause = "robot" if "robot" in e["touch"] else "held" if (e["held"] and "held" in e["touch"]) else \
                "released" if ("other_object" in e["touch"] or "held" in e["touch"]) else "shake"
        R.setdefault(c, {})[k] = dict(s=s, safe=s and not col, col=col, box=h["box_moved_1mm"], cause=cause, steps=d["steps"],
                                      look=d.get("look"))
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
    rr = [R[c][k] for k in keys]; succ = [r["steps"] for r in rr if r["s"]]
    print(f"{c:8s} TSR {sum(r['s'] for r in rr):2d} SSR {sum(r['safe'] for r in rr):2d} CAR {sum(not r['col'] for r in rr):2d} "
          f"box {sum(r['box'] for r in rr):2d} ETS {sum(succ)/max(len(succ),1):.0f} causes {dict(collections.Counter(r['cause'] for r in rr if r['box']))}")
    L = [r["look"] for r in rr if r["look"]]
    if L:
        print(f"         look: decisions {sum(l['n'] for l in L)}, switched {sum(l['switched'] for l in L)}, "
              f"nominal unsafe {sum(l['nominal_unsafe'] for l in L)}, all 8 unsafe {sum(l['all_unsafe'] for l in L)}")
for a, b in (("OBJLOOK", "C0"), ("OBJ", "VIS"), ("LOOK", "C0"), ("OBJLOOK", "LOOK")):
    for f, nm in (("safe", "SSR"), ("s", "TSR")):
        n10, n01, p, lo, hi = mc(a, b, f)
        print(f"{a} vs {b} {nm}: only {a} {n10}, only {b} {n01}, p={p:.3g}, 95% CI [{lo:+.1f}, {hi:+.1f}] pt")
