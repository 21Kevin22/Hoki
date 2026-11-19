"""Section 3/7: sha256 of every file under the given paths + git commit + pip freeze -> MANIFEST.json.
Usage: python3 scripts/make_manifest.py task/ scripts/ <weights dir> ... [--out MANIFEST.json]
"""
import hashlib, json, pathlib, subprocess, sys, datetime
args = sys.argv[1:]; out = "MANIFEST.json"
if "--out" in args:
    i = args.index("--out"); out = args[i + 1]; args = args[:i] + args[i + 2:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
files = {}
for a in args:
    p = pathlib.Path(a)
    for f in ([p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts)):
        files[str(f)] = sha(f)
run = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
m = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
         git_commit=run("git rev-parse HEAD"), git_dirty=bool(run("git status --porcelain")),
         python=sys.version.split()[0], pip_freeze=run(f"{sys.executable} -m pip freeze").splitlines(),
         files=files)
pathlib.Path(out).write_text(json.dumps(m, indent=2))
print(f"{len(files)} files -> {out}")
