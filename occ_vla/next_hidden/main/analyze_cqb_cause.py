"""PREREG_main.md 2026-11-23 (7): classify why bystanders moved under CQB (diagnostic runs main/runs/CQB_cause).
Each displacement jump is assigned to ONE cause by priority: robot (arm/hand/finger) > held object > other task object
(target not held / dest / container) > another bystander (chain) > nothing touching (falls / settles).
Reported per cause: total jump (mm), episodes where it contributed > 1 mm, and the share of jump mass with an active fence."""
import json, glob, collections, sys
PRI = [("robot", {"arm", "hand", "finger"}), ("held", {"held"}), ("task_obj", {"target", "dest", "container"}),
       ("chain", {"bystander"}), ("table_only", {"table"})]


def cause(touch):
    for k, s in PRI:
        if set(touch) & s:
            return k
    return "nothing"


d = sys.argv[1] if len(sys.argv) > 1 else "main/runs/CQB_cause"
mass, fenced, ep_n, n_unsafe, n = collections.Counter(), collections.Counter(), collections.Counter(), 0, 0
held_phase = collections.Counter()
for f in sorted(glob.glob(f"{d}/*.json")):
    j = json.load(open(f)); n += 1
    c = j["collision"]
    if not c["safelibero_1mm"]:
        continue
    n_unsafe += 1
    per = collections.Counter()
    for e in [e for e in c["cause"] if e["t"] >= 10]:   # after the settle steps (rebase at t = 10)
        k = cause(e["touch"]); per[k] += e["jump"]; mass[k] += e["jump"]
        if e["fenced"]:
            fenced[k] += e["jump"]
        held_phase[(k, bool(e["held"]))] += e["jump"]
    for k, v in per.items():
        if v > 0.001:
            ep_n[k] += 1
    print(f.split("/")[-1], "succ" if j["success"] else "FAIL", {k: round(v * 1000, 1) for k, v in per.most_common()})
print(f"\n{n} episodes, {n_unsafe} moved a bystander > 1 mm")
for k, _ in PRI + [("nothing", None)]:
    print(f"{k:10} total {mass[k]*1000:8.1f} mm  episodes>1mm {ep_n[k]:3}  fenced share {fenced[k]/mass[k] if mass[k] else 0:.2f}"
          f"  while holding {held_phase[(k, True)]/mass[k] if mass[k] else 0:.2f}")
