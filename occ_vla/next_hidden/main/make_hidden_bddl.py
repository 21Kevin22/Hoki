"""PREREG_main.md 2026-11-24 (8) step 1: hidden-obstacle variants of trained LIBERO tasks.
Copies the task's BDDL and adds ONE small object (butter_9, a LIBERO 'butter' box) parked on the table; instruction,
layout and goal are unchanged. The object is moved into the container (drawer / microwave) after reset by
main/hidden_initstates.py (LIBERO's sampler cannot place objects inside closed drawers).
Usage: python main/make_hidden_bddl.py --task H1 -> main/bddl_hidden/H1.bddl"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import bddl_path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# name: (suite, index, table fixture, parking (x, y) on the table)
HIDDEN = {
    "H1": ("libero_goal", 3, "main_table", (0.18, 0.10)),     # open the top drawer and put the bowl inside
    "H2": ("libero_10", 3, "kitchen_table", (0.18, -0.20)),   # black bowl -> bottom drawer, close it
    "H3": ("libero_10", 9, "kitchen_table", (0.18, 0.12)),    # mug -> microwave, close it
    "H5": ("libero_goal", 0, "main_table", (0.18, 0.10)),     # open the middle drawer (over-stopping check)
    "H4": ("clutter:T5_k2", None, "study_table", (0.20, -0.15)),  # book -> caddy, our k2 clutter; box behind a tall object
}
OBJ, CLS = "butter_9", "butter"


def make(task):
    suite, idx, table, (x, y) = HIDDEN[task]
    src = (os.path.join(HERE, "main", "bddl_clutter", suite.split(":")[1] + ".bddl") if suite.startswith("clutter:")
           else bddl_path(suite, idx)[0])
    s = open(src).read()
    reg = f"""      (hidden_park_region
          (:target {table})
          (:ranges (
              ({x - 0.01:.3f} {y - 0.01:.3f} {x + 0.01:.3f} {y + 0.01:.3f})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
"""
    s = re.sub(r"(\(:regions\s*\n)", lambda m: m.group(1) + reg, s, count=1)
    s = re.sub(r"(\(:objects\s*\n)", lambda m: m.group(1) + f"    {OBJ} - {CLS}\n", s, count=1)
    s = re.sub(r"(\(:init\s*\n)", lambda m: m.group(1) + f"    (On {OBJ} {table}_hidden_park_region)\n", s, count=1)
    out = os.path.join(HERE, "main", "bddl_hidden"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"{task}.bddl"); open(p, "w").write(s)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); a = ap.parse_args()
    print(make(a.task))
