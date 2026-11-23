"""Stage-1 / stage-2 analysis of the failure-cause study (PREREG_main.md 2026-11-23 (2), (3)). Rules fixed there:
collision event = an L2 step (disallowed contact > 5 N) or a bystander / the container moved >= 1 cm unintentionally;
occlusion event = a target not held, or a destination / the container, with agentview occlusion ratio >= 0.5 AND < 10 % of
its points visible in the wrist camera, for >= 20 consecutive steps.  Usage: python main/analyze_attrib.py B1_attrib"""
import glob, json, os, sys
from collections import Counter, defaultdict


def events(r):
    a = r["collision"]["attrib"]
    col = any(a["l2"]) or max(a["byst"] + [0]) >= 0.01 or max(a["cont"] + [0]) >= 0.01
    occ, run = False, [0] * len(a["objects"])
    for t in range(len(a["occ_ag"])):
        held = set(a["held"][t])
        for i, (lab, kind) in enumerate(a["objects"]):
            on = a["occ_ag"][t][i] >= 0.5 and a["vis_wr"][t][i] < 0.10 and not (kind == "target" and lab in held)
            run[i] = run[i] + 1 if on else 0
            if run[i] >= 20:
                occ = True
    return col, occ


def label(col, occ):
    return {(True, True): "both", (True, False): "collision", (False, True): "occlusion", (False, False): "neither"}[(col, occ)]


def main(d):
    per = defaultdict(list)
    for f in sorted(glob.glob(f"main/runs/{d}/*.json")):
        r = json.load(open(f))
        task = os.path.basename(f).split("_diag")[0]
        per[task].append((r["success"], label(*events(r)), os.path.basename(f)))
    tot = {True: Counter(), False: Counter()}
    print(f"{'task':8s} n  succ  | failures: collision occlusion both neither | successes: collision occlusion both neither")
    for t in sorted(per, key=lambda x: (len(x), x)):
        L = per[t]; s = sum(x[0] for x in L)
        cf = Counter(l for ok, l, _ in L if not ok); cs = Counter(l for ok, l, _ in L if ok)
        tot[False].update(cf); tot[True].update(cs)
        print(f"{t:8s} {len(L):2d} {s:3d}   |  " + "  ".join(f"{cf[k]:3d}" for k in ("collision", "occlusion", "both", "neither"))
              + "        |   " + "  ".join(f"{cs[k]:3d}" for k in ("collision", "occlusion", "both", "neither")))
    for ok in (False, True):
        n = sum(tot[ok].values())
        print(("successes" if ok else "failures "), n, {k: f"{v} ({100 * v / max(n, 1):.0f}%)" for k, v in tot[ok].items()})
    return per


if __name__ == "__main__":
    main(sys.argv[1])
