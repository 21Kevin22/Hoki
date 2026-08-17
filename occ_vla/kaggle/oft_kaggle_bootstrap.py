# %% [markdown]
# # oft_kaggle_bootstrap
#
# Cell-by-cell setup + smoke test for running the OpenVLA-OFT
# occlusion-recovery harness (`thirdparty/openvla-oft/` +
# `scripts/run_oft_camera_dropout_eval.py`) on a Kaggle Notebook GPU
# instance, since the actual project server is unreachable this session.
#
# **This file is meant to be uploaded to Kaggle directly as a notebook**
# (`kaggle/oft_kaggle_bootstrap.ipynb`, generated from this .py via
# `jupytext --to notebook`, sits next to this file -- upload THAT one).
# Run cells top to bottom once; on later sessions, re-run only the smoke
# test near the bottom once setup is cached in a Kaggle Dataset (see the
# "persist" cell).
#
# **Revision history (real Kaggle runs, 2026-08-18)**, most recent first:
#
# 4. **`import libero` "succeeded" but was actually broken, two layers
#    deep.** First: `pip install -e LIBERO_DIR` (default PEP 660 editable
#    mode) reported success, but `import libero` raised
#    `ModuleNotFoundError` -- `pip show -f libero` showed the editable
#    finder files WERE created, but a `sys.path` dump showed only
#    openvla-oft's own finder hook was actually registered at interpreter
#    startup, not LIBERO's -- its `.pth`/finder silently never loaded.
#    Root cause: LIBERO's nested `libero/libero/` package layout (the repo
#    root's `libero/` dir is not itself the importable package; the real
#    one is one level deeper) trips up the modern build backend's
#    auto-discovery. Fixed with `--config-settings editable_mode=compat`
#    (forces the classic `setup.py`-driven editable install, which
#    respects the layout `setup.py` explicitly declares). Second, once
#    THAT was fixed, `import libero` succeeded but `libero.__file__` was
#    `None` (an empty PEP 420 namespace package) -- turned out to be
#    harmless: `from libero.libero import benchmark` (what
#    `run_oft_camera_dropout_eval.py` actually needs) resolved to a real
#    file correctly. Third, THAT import surfaced a real, separate problem:
#    a `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`
#    from torch, because `libero_requirements.txt`'s install had silently
#    upgraded numpy to 2.2.6 (incompatible with `torch==2.2.0`'s compiled
#    C-extension, which needs NumPy's 1.x C-API) and mujoco to 3.11.0
#    (incompatible with `robosuite==1.4.1`'s `mj_fullM()` signature, an
#    already-known cross-track finding in this project). Fixed by
#    re-pinning both explicitly, AFTER the requirements install, in cell 3
#    (see its own revision note there).
# 3. **`No space left on device` even after freeing everything findable
#    under `/kaggle/working`** -- `df -h` (real output pasted into this
#    revision note) showed the actual container filesystem (`overlay`,
#    mounted at `/`) had **1.1TB free**, while `shutil.disk_usage`
#    scoped to `/kaggle/working` reported only ~11GB free -- Kaggle
#    enforces a separate, much smaller soft quota (commonly ~20GB) SPECIFIC
#    to `/kaggle/working` (since that directory's contents get versioned
#    as the notebook's saved output), not a real disk-space shortage.
#    **Fix: every heavy path (venv, git clones, checkpoint, logs) now
#    lives under `WORK_ROOT = "/root/oft_work"`, outside `/kaggle/working`
#    entirely**, using the 1.1TB pool instead of the ~20GB one. Tradeoff:
#    `/root/oft_work` is NOT auto-saved by Kaggle's "Save Version" the way
#    `/kaggle/working` is -- irrelevant here since checkpoint persistence
#    already goes through a manual Kaggle Dataset upload (cell 6), not
#    through `/kaggle/working`'s own versioning.
# 2. `snapshot_download` failed with `RuntimeError: ... Background writer
#    channel closed` -- huggingface_hub's newer "Xet Storage" fast-download
#    backend (`hf_xet`) is flaky in Kaggle's sandboxed filesystem. Attempted
#    fix: `HF_HUB_DISABLE_XET=1` + `pip uninstall hf_xet` before the first
#    huggingface_hub call. **Only partially effective** -- a later run's
#    traceback still showed `xet_get(...)` firing, so cell 5b now verifies
#    `hf_xet`'s import status directly rather than assuming the disable
#    worked; if it's still importable, that's a live loose end (the actual
#    failure blocking runs at this point in the story turned out to be the
#    disk quota above, not this, so it wasn't chased further).
# 1. `pip install -e thirdparty/openvla-oft` failed with `CalledProcessError`
#    and NO visible error text, because the original version of this
#    notebook used `subprocess.run(..., check=True)` without
#    `capture_output` -- fixed via the `run()` helper (cell 0), which
#    always prints stdout/stderr before raising, everywhere in this
#    notebook from here on. Suspected (not fully confirmed) root cause of
#    the original failure: Kaggle's base kernel is Python 3.12, but
#    openvla-oft's `SETUP.md` calls for `python=3.10` and pins several old
#    exact dependency versions unlikely to have 3.12 wheels -- fixed by
#    creating an isolated Python 3.10 venv (`VENV_PY`, cell 2) and running
#    every environment-sensitive step through it, never the notebook
#    kernel's own Python.
#
# Kaggle-specific constraints this notebook is written around (see the
# occ_vla/CLAUDE.md discussion, 2026-08-17):
# - Sessions are ephemeral: everything is wiped when the session ends
#   regardless of which directory it's under. Persist the downloaded
#   checkpoint to a private Kaggle Dataset (see the "persist" cell) so
#   later sessions don't re-pay the ~15GB download + dependency install
#   every time.
# - GPU memory is 16GB (T4x2 or P100) -- OpenVLA-OFT is 7B params, ~14GB
#   in bf16 alone. Leaves little headroom for LIBERO's rendering + the
#   vjepa predictor's activations; consider `load_in_4bit` if this OOMs.
# - GPU quota is ~30h/week, shared across T4/P100 -- budget accordingly.
#   The smoke test near the bottom is deliberately small (num-trials=1,
#   4 conditions) so a first successful run costs minutes, not hours.
# - `/kaggle/working` specifically has its own separate, much smaller
#   quota (~20GB) than the container's real disk (~1.1TB free observed) --
#   see revision note 3 above. This notebook now keeps everything heavy
#   OUTSIDE `/kaggle/working` for that reason.

# %% [markdown]
# ## 0. Helper: a subprocess wrapper that never hides its own error text
#
# Every cell below uses this instead of bare `subprocess.run(..., check=True)`
# -- always prints stdout/stderr BEFORE raising, so a failure is always
# debuggable from the cell output alone (see revision note 1 above).
#
# **Revision note (2026-08-18, revision 5)**: also strips/overrides
# `MPLBACKEND` in the child process's environment. Jupyter/IPython sets
# `MPLBACKEND=module://matplotlib_inline.backend_inline` in the KERNEL's
# own environment (for inline plot rendering) -- `subprocess.run` inherits
# the full parent environment by default, so every `VENV_PY` subprocess
# call was getting this value too, and matplotlib's own `rcParams`
# validation rejects it outside an actual IPython context (a real
# `ValueError` observed crashing `run_oft_camera_dropout_eval.py`, which
# imports matplotlib transitively via LIBERO's env wrapper). Forced to
# `Agg` (a plain headless raster backend, appropriate here regardless --
# there's no display) for every subprocess this helper launches.

# %%
import subprocess


def run(cmd, cwd=None, check=True):
    import os

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    print("$ " + " ".join(cmd) + (f"   (cwd={cwd})" if cwd else ""))
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print(result.stderr[-4000:])
    if check and result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}: {' '.join(cmd)}")
    return result


# %% [markdown]
# ## 1. Environment check
# Confirms a real GPU + EGL headless rendering are actually available
# BEFORE spending time on the (slow) dependency install below. If EGL
# isn't available, LIBERO's offscreen rendering will not work and nothing
# past this cell is worth running.
#
# Also sets `WORK_ROOT` -- see revision note 3 above: deliberately OUTSIDE
# `/kaggle/working`, which has its own much smaller (~20GB) quota separate
# from the container's real disk.

# %%
import os

WORK_ROOT = "/root/oft_work"
os.makedirs(WORK_ROOT, exist_ok=True)

print(run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], check=False).stdout)

import ctypes

try:
    ctypes.CDLL("libEGL.so.1")
    print("libEGL.so.1 loads OK -- MUJOCO_GL=egl should work")
except OSError as e:
    print(f"libEGL NOT found ({e}) -- may need `apt-get install -y libegl1` below, "
          "or fall back to MUJOCO_GL=osmesa (slower, CPU-side software rendering)")

import shutil

print(run(["df", "-h"], check=False).stdout)  # look at every mount -- /kaggle/working's own quota is NOT the whole story, see revision note 3
print(f"Disk free under {WORK_ROOT} (GB):", shutil.disk_usage(WORK_ROOT).free / 1e9)

# %% [markdown]
# ## 2. System packages (EGL headless rendering + an isolated Python 3.10)
#
# openvla-oft's own `SETUP.md` calls for `python=3.10`; Kaggle's default
# notebook kernel is Python 3.12 and several of openvla-oft's exact-pinned
# old dependencies likely lack 3.12 wheels. Rather than fight that in the
# notebook's own kernel, install python3.10 + venv alongside it and run
# every openvla-oft/LIBERO-touching command through THAT interpreter --
# `VENV_PY`, defined here, is reused by every cell below.

# %%
# Kaggle images are Debian-based -- apt works without sudo as root.
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "-qq", "libegl1", "libgl1", "libosmesa6-dev", "patchelf",
     "python3.10", "python3.10-venv", "python3.10-dev"])

# %%
VENV_DIR = os.path.join(WORK_ROOT, "venv_oft")
if not os.path.isdir(VENV_DIR):
    run(["python3.10", "-m", "venv", VENV_DIR])
VENV_PY = os.path.join(VENV_DIR, "bin", "python")
run([VENV_PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])
print(run([VENV_PY, "--version"], check=False).stdout)

# %% [markdown]
# ## 3. Clone the repo + install dependencies
#
# Two SEPARATE environments are needed per thirdparty/openvla-oft/CLAUDE.md
# ("Environment setup (new server / fresh clone)"): the OFT/LIBERO stack
# here (now `VENV_PY`, Python 3.10), plus (only if you also want the
# pi0.5/MMaDA track's own scripts, which this smoke test does NOT need) a
# completely separate stack -- don't install both into the same env, they
# conflict (different pinned torch/transformers/mujoco versions).

# %%
REPO_DIR = os.path.join(WORK_ROOT, "Hoki")
if not os.path.isdir(REPO_DIR):
    run(["git", "clone", "https://github.com/21Kevin22/Hoki.git", REPO_DIR])
OCC_VLA_DIR = os.path.join(REPO_DIR, "occ_vla")
OFT_DIR = os.path.join(OCC_VLA_DIR, "thirdparty", "openvla-oft")

# %%
# openvla-oft's own pinned deps (see thirdparty/openvla-oft/pyproject.toml)
# -- note the custom transformers fork (bidirectional attention for
# parallel decoding), NOT stock `transformers` from PyPI. If this fails,
# the ACTUAL error text will print above this cell's RuntimeError (see the
# `run()` helper in cell 0).
run([VENV_PY, "-m", "pip", "install", "-e", OFT_DIR], cwd=OFT_DIR)

# %%
# LIBERO itself, per thirdparty/openvla-oft/LIBERO.md -- a SEPARATE clone
# from Lifelong-Robot-Learning/LIBERO (distinct from the pi0.5 track's own
# openpi-submodule LIBERO copy; don't mix the two up). Installed into the
# SAME venv_oft interpreter as openvla-oft above.
LIBERO_DIR = os.path.join(WORK_ROOT, "LIBERO")
if not os.path.isdir(LIBERO_DIR):
    run(["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", LIBERO_DIR])
# Revision note (2026-08-18): plain `pip install -e LIBERO_DIR` builds a
# PEP 660 editable install whose finder never actually got registered for
# LIBERO's nested `libero/libero/` package layout (`import libero`
# succeeded as an empty PEP 420 namespace package, `libero.__file__ is
# None`, but `libero.libero`'s real finder was silently missing from
# sys.path). `--config-settings editable_mode=compat` forces the classic
# setup.py-driven editable install instead, which handles this nested
# layout correctly (confirmed on real Kaggle infra this session).
run([VENV_PY, "-m", "pip", "install", "-e", LIBERO_DIR, "--config-settings", "editable_mode=compat"])
run([VENV_PY, "-m", "pip", "install", "-r",
     os.path.join(OFT_DIR, "experiments/robot/libero/libero_requirements.txt")])

# Revision note (2026-08-18): the requirements install above pulled in
# numpy==2.2.6 and mujoco==3.11.0, BOTH confirmed-broken on real Kaggle
# infra this session: (1) torch==2.2.0's compiled C-extension needs
# NumPy's 1.x C-API ("Failed to initialize NumPy: _ARRAY_API not found",
# a real UserWarning observed, not hypothetical) -- pip's own resolver even
# printed this exact conflict at install time
# ("tensorflow 2.15.0 requires numpy<2.0.0,>=1.23.5, but you have
# numpy 2.2.6"), it just doesn't refuse to proceed. (2) mujoco>=3.1 breaks
# robosuite==1.4.1's `mj_fullM()` call signature -- an already-documented
# cross-track finding in this project (pi0.5/MWM work), now independently
# confirmed relevant to this track's own dependency resolution too. Pin
# both back down explicitly, AFTER the requirements install so this wins
# the last-writer-takes-it fight against pip's own resolution.
run([VENV_PY, "-m", "pip", "install", "-q", "numpy==1.26.4", "mujoco==3.0.0"])

# Revision note (2026-08-18): --load-in-4bit (needed -- the 7B model OOMs
# in bf16 on a 16GB Kaggle T4) needs bitsandbytes, not in either
# requirements file. Plain `pip install bitsandbytes` (unpinned) silently
# upgraded torch 2.2.0 -> 2.13.0 to satisfy the latest bitsandbytes'
# `torch<3,>=2.4` requirement, breaking torchvision==0.17.0
# ("operator torchvision::nms does not exist"). bitsandbytes==0.43.1 is a
# version contemporary with torch 2.2.0 (pre-dates its `torch>=2.4`
# requirement) -- confirmed working on real Kaggle infra this session.
run([VENV_PY, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes==0.43.1"])

# Revision note (2026-08-18): quantized model loading
# (AutoModelForVision2Seq.from_pretrained(..., quantization_config=
# BitsAndBytesConfig(...), device_map=...)) raised
# "`.to` is not supported for `4-bit`/`8-bit` bitsandbytes models" with
# EVERY device_map tried that resolves to a single device (None,
# {"": torch.device}, {"": 0}, "auto"+max_memory forcing one GPU) --
# confirmed on real Kaggle infra this session, five separate device_map/
# quantization-API permutations, all failing identically. Root cause:
# pip's default resolution pulled `accelerate==1.14.0` (a major-version
# rewrite of the library), while openvla-oft's pinned transformers fork
# dates to ~mid-2024, contemporary with accelerate's 0.3x line --
# downgrading accelerate (NOT changing device_map/quantization_config
# again) is what actually fixed it. --load-in-4bit end-to-end confirmed
# working after this pin: a real episode completed with success=True.
run([VENV_PY, "-m", "pip", "install", "-q", "accelerate==0.30.1"])

# %%
# huggingface_hub is used below by the (environment-agnostic) checkpoint
# download cell, which intentionally stays on the notebook's OWN kernel
# (3.12) -- it doesn't touch torch/transformers/LIBERO at all, so it
# doesn't need venv_oft. Just make sure it's present here too.
run(["pip", "install", "-q", "huggingface_hub"])

# %% [markdown]
# ## 4. LIBERO first-import config (avoids an interactive hang)
# `libero/libero/__init__.py` prompts for a dataset path on first import
# and blocks on stdin in a non-interactive kernel -- pre-seed the config
# file before anything imports `libero.libero`. Written by hand (no
# `pyyaml` import) to avoid yet another package needing to be present in
# BOTH the notebook kernel and venv_oft. Already outside `/kaggle/working`
# (`~/.libero_oft` == `/root/.libero_oft` since notebooks run as root).

# %%
import pathlib

libero_config_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft")))
libero_config_dir.mkdir(parents=True, exist_ok=True)
(libero_config_dir / "config.yaml").write_text(
    f"benchmark_root: {libero_config_dir / 'benchmark'}\n"
    f"bddl_files: {pathlib.Path(LIBERO_DIR) / 'libero/libero/bddl_files'}\n"
    f"init_states: {pathlib.Path(LIBERO_DIR) / 'libero/libero/init_files'}\n"
    f"datasets: {libero_config_dir / 'datasets'}\n"
    f"assets: {pathlib.Path(LIBERO_DIR) / 'libero/libero/assets'}\n"
)
os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_dir)
print("Wrote", libero_config_dir / "config.yaml")

# %% [markdown]
# ## 5. Download ONE checkpoint (~15GB) -- pick the suite you're testing
#
# **Critical, per thirdparty/openvla-oft/CLAUDE.md**: download to a LOCAL
# DIRECTORY, never pass a bare HF repo ID as `--checkpoint` -- the
# occlusion-recovery code (`vjepa_latent_predictor.py`) only gets wired
# into the checkpoint's config on first load from a local dir; a bare repo
# ID silently loads the STOCK model with zero occlusion-recovery capability
# and no error/warning. See revision notes 2-3 above for what already went
# wrong here and what this version changes (Xet disable attempt +
# WORK_ROOT instead of /kaggle/working).

# %%
os.environ["HF_HUB_DISABLE_XET"] = "1"
run(["pip", "uninstall", "-y", "-q", "hf_xet"], check=False)  # ok if it wasn't installed at all

try:
    import hf_xet  # noqa: F401

    print("hf_xet IS still importable -- the uninstall above did not fully remove it (known loose end, see revision note 2)")
except ImportError:
    print("hf_xet is not importable now -- the Xet code path should be genuinely unavailable")

print(f"Disk free under {WORK_ROOT} before download (GB):", shutil.disk_usage(WORK_ROOT).free / 1e9)

# %%
from huggingface_hub import snapshot_download

CHECKPOINT_DIR = os.path.join(WORK_ROOT, "checkpoints", "openvla-7b-oft-libero10-vjepa")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
snapshot_download(
    repo_id="moojink/openvla-7b-oft-finetuned-libero-10",  # swap suite as needed: -spatial / -object / -goal
    local_dir=CHECKPOINT_DIR,
    max_workers=4,  # lower than the default if this still errors intermittently -- the earlier Xet crash was concurrency-adjacent
    force_download=True,  # don't try to resume/verify against partial state a prior attempt left behind
)
print("Downloaded to", CHECKPOINT_DIR)

# %% [markdown]
# ## 6. Persist to a Kaggle Dataset (do this ONCE, reuse across sessions)
#
# Run this cell, then in the Kaggle UI: "Save Version" the resulting
# dataset (Kaggle's own `kagglehub`/dataset-upload flow, not scriptable
# reliably from inside a notebook without your own API token configured --
# do this step through the Kaggle website's "New Dataset" upload UI
# pointed at the printed CHECKPOINT_DIR, OR use the `kaggle datasets
# create`/`version` CLI if you have `~/.kaggle/kaggle.json` configured).
# Next session: skip cells 3-5 entirely and just
# `!kaggle datasets download -d <your-username>/oft-libero10-vjepa-ckpt`.
# (Note this directory lives under WORK_ROOT / `/root`, NOT
# `/kaggle/working` -- Kaggle's own automatic "Save Version" output-saving
# does NOT cover it; this manual dataset-upload step is the only way it
# survives past this session, not an optional nicety.)

# %%
print(
    "Manual step: upload", CHECKPOINT_DIR, "as a private Kaggle Dataset via "
    "the website UI (Datasets -> New Dataset -> upload folder), or "
    f"`kaggle datasets create -p {os.path.dirname(CHECKPOINT_DIR)} --dir-mode zip` "
    "if the CLI is configured. This avoids re-downloading ~15GB every session."
)

# %% [markdown]
# ## 7. Smoke test: the extended logging + debounce-gate pipeline (2026-08-17/18)
#
# Small on purpose (num-trials=1, 4 conditions incl. the two new ones --
# `wrist_partial_vjepa_gated`, `wrist_partial_prevframe`) so this costs
# minutes of GPU quota, not hours. Exercises:
#   - per-step JSONL logging (StepLogWriter -- oft_step_logger.py)
#   - the debounce gate (OcclusionGate -- oft_occlusion_gate.py), both
#     live-gated (*_gated condition) and post-hoc-recomputable from the
#     same S_occ log (B1)
#   - occ_gt (patch-level ground truth -- oft_occlusion_gt.py)
#   - A1 latency timing (--measure-latency)
#   - B3's zero-parameter previous-frame-copy control
#   - `--load-in-4bit` (REQUIRED -- the 7B model OOMs in plain bf16 on a
#     16GB Kaggle T4; confirmed end-to-end working after the accelerate
#     pin above, real episode completed with success=True)
#
# Runs via `VENV_PY` (Python 3.10), not the notebook kernel -- this is the
# actual point of cell 2's venv setup. Uses a live-streaming subprocess
# call (not the `run()` helper, which buffers ALL output until the process
# exits) -- a real multi-episode LIBERO rollout with a 7B model can take
# several minutes with ZERO output in between (this script only prints
# once per COMPLETED episode), which looked indistinguishable from a hang
# using `run()` on real Kaggle infra this session. If this cell runs
# clean, the pipeline is validated end-to-end for real (not just
# unit-tested on a Mac with no GPU/LIBERO) -- THEN scale up --num-trials
# for the real A1/A2/A4/B1/B3 runs.

# %%
import subprocess

STEPLOGS_DIR = os.path.join(WORK_ROOT, "steplogs_smoketest")
RESULTS_PATH = os.path.join(WORK_ROOT, "smoketest_results.json")

cmd = [
    VENV_PY, "scripts/run_oft_camera_dropout_eval.py",
    "--task-suite", "libero_10",
    "--task-id", "8",  # moka_pots -- this project's most-tested task
    "--num-trials", "1",
    "--checkpoint", CHECKPOINT_DIR,
    "--load-in-4bit",
    "--conditions", "baseline", "wrist_partial", "wrist_partial_vjepa_gated", "wrist_partial_prevframe",
    "--log-steps-dir", STEPLOGS_DIR,
    "--s-occ-source", "oracle",
    "--debounce-k", "3",
    "--measure-latency",
    "--results-path", RESULTS_PATH,
]
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["PYTHONUNBUFFERED"] = "1"

proc = subprocess.Popen(cmd, cwd=OCC_VLA_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print("\nExit code:", proc.returncode)

# %% [markdown]
# ## 8. Inspect one step log by hand before trusting anything downstream
# Per this project's own standing discipline (occ_vla CLAUDE.md,
# "visually/manually inspect before trusting a metric") -- print a few raw
# rows and sanity-check them: does s_occ/occ_gt look like a real fraction
# in [0,1]? does correction_applied only ever go True after k=3 consecutive
# occ_flag=True rows on the *_gated condition? does t_vla_ms look like a
# plausible number of milliseconds (not 0 everywhere, not absurdly huge)?

# %%
import json

log_path = os.path.join(STEPLOGS_DIR, "wrist_partial_vjepa_gated_ep000.jsonl")
with open(log_path) as f:
    rows = [json.loads(line) for line in f]
for row in rows[:15]:
    print(row)
