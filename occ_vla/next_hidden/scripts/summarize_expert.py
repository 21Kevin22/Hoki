"""Summarize results/expert/all_states.jsonl into results/expert/summary.json and print a table."""
import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "expert", "all_states.jsonl"))]
tab = []
for s in rows:
    p = s["phases"]
    tab.append(dict(state=s["state"], success=s["success"], steps=s["steps"], door_deg=p["door_deg_after_pull"],
                    door_open=p["door_open_predicate"], entrance_m=p["entrance_width_m"], handle_reached=p["at_handle"],
                    mug_out=p["mug_out_of_region"], mug_plate_xy_m=p["final_mug_plate_xy_m"],
                    handle_xyz=p["handle_world"], first_visible=s["first_visible_step"],
                    robot_microwave_contact_steps=sum(s["robot_microwave_contact_steps"].values()),
                    robot_microwave_contact_by_phase={k: v for k, v in s["robot_microwave_contact_steps"].items() if v},
                    robot_other_object_contact_steps=s["robot_other_object_contact_steps"]))
out = dict(n=len(tab), n_success=sum(t["success"] for t in tab),
           entrance_width_vs_door_deg=rows[0]["phases"]["entrance_width_at_deg"], rows=tab)
json.dump(out, open(os.path.join(HERE, "results", "expert", "summary.json"), "w"), indent=1)
for t in tab:
    print(t["state"], t["success"], t["steps"], t["door_deg"], t["handle_reached"], t["mug_plate_xy_m"], t["first_visible"],
          t["robot_microwave_contact_by_phase"], t["robot_other_object_contact_steps"])
print("success", out["n_success"], "/", out["n"], "width vs door deg", out["entrance_width_vs_door_deg"])
