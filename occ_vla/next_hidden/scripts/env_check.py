"""Section 4: environment check (no GPU work). Writes results/env_report.json.
Run:  python3 scripts/env_check.py
"""
import json, os, shutil, subprocess, sys, datetime, pathlib

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

def cgroup(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except Exception:
        return "unknown"

gpus = []
for line in sh("nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,utilization.gpu "
               "--format=csv,noheader,nounits").splitlines():
    i, name, cc, tot, used, util = [x.strip() for x in line.split(",")]
    major = int(cc.split(".")[0])
    gpus.append(dict(index=int(i), name=name, compute_cap=cc,
                     generation="Ampere+" if major >= 8 else "Turing-or-older",
                     mem_total_mib=int(tot), mem_used_mib=int(used), util_pct=int(util),
                     mem_free_mib=int(tot) - int(used)))
procs = sh("nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader")
smi = sh("nvidia-smi")
driver = next((l for l in smi.splitlines() if "Driver Version" in l), "")

mem_max = cgroup("/sys/fs/cgroup/memory.max")
report = dict(
    timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    gpus=gpus,
    gpu_compute_processes=procs or "none",
    driver_line=driver.strip("| "),
    logged_in_users=sh("who | awk '{print $1}' | sort -u | tr '\\n' ' '"),
    other_users_processes=sh("ps -eo user= | sort -u | grep -v -E '^(root|ubuntu|systemd.*|messagebus|syslog|_.*|nobody|daemon)$' | tr '\\n' ' '"),
    host_mem_total_gib=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
    cgroup_memory_max=(f"{int(mem_max)/2**30:.1f} GiB" if mem_max.isdigit() else mem_max),
    cgroup_cpu_max=cgroup("/sys/fs/cgroup/cpu.max"),
    cpu_cores=os.cpu_count(),
    disk_free_gib=round(shutil.disk_usage(os.path.expanduser("~")).free / 2**30, 1),
    dtype_policy=("bf16 OK (all GPUs Ampere+)" if gpus and all(g["generation"] == "Ampere+" for g in gpus)
                  else "Turing present: test pi0.5 fp32/fp16, Qwen fp16 + sdpa, no FlashAttention 2"),
    openpi_present=any(pathlib.Path(p).exists() for p in [
        os.path.expanduser("~/slocal1/third_party/openpi"),
        os.path.expanduser("~/slocal1/Hoki/occ_vla/third_party/openpi")]),
    pi05_libero_checkpoint=sh("find ~/.cache -maxdepth 4 -type d -name 'pi05_libero*' 2>/dev/null") or "not found",
    qwen_checkpoint=sh("find ~/.cache/huggingface -maxdepth 3 -type d -iname '*Qwen2.5-VL-3B*' 2>/dev/null") or "not found",
)

# Rendering check: MuJoCo EGL offscreen (same backend to be used for all experiments)
render = sh(f"MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES='' {sys.executable} -c "
            "\"import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\\\"0 0 3\\\"/>"
            "<geom type=\\\"box\\\" size=\\\".1 .1 .1\\\"/></worldbody></mujoco>');d=mujoco.MjData(m);"
            "r=mujoco.Renderer(m,64,64);mujoco.mj_forward(m,d);r.update_scene(d);"
            "print('ok',mujoco.__version__,float(r.render().mean()))\"")
report["render_backend"] = "egl"
report["render_check"] = render or "FAILED (empty output)"
report["python"] = sys.executable + " " + sys.version.split()[0]

out = pathlib.Path(__file__).resolve().parents[1] / "results" / "env_report.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
