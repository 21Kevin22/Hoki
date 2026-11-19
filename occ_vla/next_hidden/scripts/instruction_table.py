"""6-3: map candidate instructions to the closest pi0.5-LIBERO training task names.
Training tasks: physical-intelligence/libero meta/tasks.jsonl (the dataset named in openpi's pi05_libero config),
saved as results/pi05_libero_train_tasks.jsonl. Similarity = token-set Jaccard and difflib ratio (both reported;
nothing here depends on a model). Writes results/instruction_table.json and prints a markdown table."""
import difflib, json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train = [json.loads(l)["task"] for l in open(os.path.join(HERE, "results", "pi05_libero_train_tasks.jsonl"))]
CANDIDATES = {
    "A (single)": ["take the yellow and white mug out of the microwave and put it on the plate",
                   "open the microwave and put the yellow and white mug on the plate"],
    "B/C step 1 (open)": ["open the microwave"],
    "B/C step 2 (take + place)": ["put the yellow and white mug on the plate",
                                  "pick up the yellow and white mug and place it on the plate"],
}
def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B)
rows = []
for cond, cands in CANDIDATES.items():
    for c in cands:
        best = sorted(train, key=lambda t: -jac(c, t))[:3]
        rows.append(dict(condition=cond, instruction=c, exact_in_training=c in train,
                         nearest=[dict(task=t, jaccard=round(jac(c, t), 2),
                                       difflib=round(difflib.SequenceMatcher(None, c, t).ratio(), 2)) for t in best]))
json.dump(dict(n_training_tasks=len(train), rows=rows), open(os.path.join(HERE, "results", "instruction_table.json"), "w"), indent=1)
print(f"training tasks: {len(train)}")
for r in rows:
    n = r["nearest"][0]
    print(f'| {r["condition"]} | {r["instruction"]} | {r["exact_in_training"]} | {n["task"]} | {n["jaccard"]} |')
kw = {k: [t for t in train if k in t] for k in ["microwave", "open", "plate", "mug", "out of", "take"]}
print(json.dumps(kw, indent=1))
