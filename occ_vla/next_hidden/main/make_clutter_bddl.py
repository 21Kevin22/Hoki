"""PREREG_main.md 2026-11-23 (3): our own clutter variants of a libero_10 task. k tall objects are added between the
agentview camera (+x side of the table) and the target objects' init regions; instruction and goal are unchanged.
Occluder j of n for a target with init-region centre (cx, cy) is placed in a 3 x 3 cm region centred at
(cx + OFFSET, cy + (j - (n - 1) / 2) * 0.07), OFFSET = 0.08 m toward the camera.
Usage: python main/make_clutter_bddl.py --task T1 --k 2   -> main/bddl_clutter/T1_k2.bddl"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS, bddl_path
from parts import TARGETS

OFFSET = 0.08                       # toward the camera; 0.13 occluded too little (pre-run check, PREREG 2026-11-23 (3))
ORDER = ["milk", "orange_juice", "ketchup", "bbq_sauce", "salad_dressing", "chocolate_pudding"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make(task, k):
    src, _ = bddl_path(*TASKS[task][:2])
    s = open(src).read()
    present = set(re.findall(r"^\s*\w+\s+-\s+(\w+)\s*$", s, re.M))
    classes = [c for c in ORDER if c not in present][:k]
    targets = []
    for t in TARGETS[task]:
        mm = re.search(r"\(On\s+" + t + r"\s+(\w+)\)", s)
        if not mm:
            continue
        ref = mm.group(1)
        for reg in re.finditer(r"\((\w+)\s*\n\s*\(:target\s+(\w+)\)\s*\n\s*\(:ranges\s*\(\s*\n\s*\(([-\d.e ]+)\)", s):
            if f"{reg.group(2)}_{reg.group(1)}" == ref:
                x1, y1, x2, y2 = map(float, reg.group(3).split())
                targets.append((t, reg.group(2), (x1 + x2) / 2, (y1 + y2) / 2))
    if not targets:
        raise SystemExit(f"{task}: no target placed in a table region")
    per = [k // len(targets) + (1 if i < k % len(targets) else 0) for i in range(len(targets))]
    regions, objects, inits, n = [], [], [], 0
    for (t, table, cx, cy), cnt in zip(targets, per):
        for j in range(cnt):
            cls = classes[n]; name = f"{cls}_9"; rname = f"occ{n}_region"
            x, y = cx + OFFSET, cy + (j - (cnt - 1) / 2) * 0.07
            regions.append(f"""      ({rname}
          (:target {table})
          (:ranges (
              ({x - 0.015:.4f} {y - 0.015:.4f} {x + 0.015:.4f} {y + 0.015:.4f})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
""")
            objects.append(f"    {name} - {cls}\n")
            inits.append(f"    (On {name} {table}_{rname})\n")
            n += 1
    s = re.sub(r"(\(:regions\s*\n)", lambda mm: mm.group(1) + "".join(regions), s, count=1)
    s = re.sub(r"(\(:objects\s*\n)", lambda mm: mm.group(1) + "".join(objects), s, count=1)
    s = re.sub(r"(\(:init\s*\n)", lambda mm: mm.group(1) + "".join(inits), s, count=1)
    out = os.path.join(HERE, "main", "bddl_clutter"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{task}_k{k}.bddl")
    open(path, "w").write(s)
    return path, classes, targets


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--k", type=int, required=True)
    a = ap.parse_args()
    print(make(a.task, a.k))
