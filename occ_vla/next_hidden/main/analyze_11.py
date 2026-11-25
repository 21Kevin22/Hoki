"""PREREG (11): C0 / CQB / CQBR on fresh stable clutter states (new2)."""
import json, glob, math, os, collections
R = {}
for c in ("C0", "CQB", "CQBR"):
    for f in glob.glob(f"main/runs/{c}_main11/*.json"):
        d = json.load(open(f)); k = os.path.basename(f)[:-5]
        s = bool(d["success"]); col = bool(d["collision"]["safelibero_1mm"])
        R.setdefault(c, {})[k] = dict(s=s, safe=s and not col, col=col, stalled=bool(d.get("stalled")), steps=d["steps"])
keys = sorted(set.intersection(*[set(v) for v in R.values()])); N = len(keys)
def mc(a, b, f, ks=None):
    ks = ks or keys; n = len(ks)
    n10 = sum(R[a][k][f] and not R[b][k][f] for k in ks); n01 = sum(R[b][k][f] and not R[a][k][f] for k in ks)
    m = n10 + n01
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(min(n10, n01) + 1)) / 2 ** m) if m else 1.0
    dl = (n10 - n01) / n; se = math.sqrt(max(((n10 + n01) / n - dl ** 2) / n, 0))
    return n10, n01, p, 100 * (dl - 1.96 * se), 100 * (dl + 1.96 * se)
print("paired n =", N)
for t in ("", "T1", "T2", "T5"):
    ks = [k for k in keys if k.startswith(t)]
    print(t or "ALL", len(ks), {c: (sum(R[c][k]["s"] for k in ks), sum(R[c][k]["safe"] for k in ks), sum(R[c][k]["col"] for k in ks), sum(R[c][k]["stalled"] for k in ks)) for c in R}, "(TSR, SSR, moved, stalled)")
for a, b, f in (("CQB", "C0", "safe"), ("CQB", "C0", "s"), ("CQBR", "CQB", "s"), ("CQBR", "CQB", "safe"), ("CQBR", "C0", "safe"), ("CQBR", "C0", "s")):
    n10, n01, p, lo, hi = mc(a, b, f)
    print(f"{a} vs {b} {'SSR' if f=='safe' else 'TSR'}: only {a} {n10}, only {b} {n01}, p={p:.3g}, CI [{lo:+.1f}, {hi:+.1f}]")
