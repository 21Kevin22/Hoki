#!/usr/bin/env python3
import os
import argparse

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env


def main():
    parser = argparse.ArgumentParser(description="List *_pos keys for LIBERO tasks")
    parser.add_argument("--suite", default="libero_10", help="Benchmark suite name")
    parser.add_argument("--resolution", type=int, default=64, help="Env image resolution")
    parser.add_argument("--mujoco_gl", default="egl", help="MUJOCO_GL backend (egl/osmesa)")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    print("num_tasks", suite.get_num_tasks())

    for tid in range(suite.get_num_tasks()):
        task = suite.get_task(tid)
        name = getattr(task, "name", None)
        desc = getattr(task, "language", None)
        init_state = suite.get_task_init_states(tid)[0]

        env, _ = get_libero_env(task, "openvla", resolution=args.resolution)
        try:
            obs = env.reset()
            obs = env.set_init_state(init_state)
            keys = sorted([k for k in obs.keys() if k.endswith("_pos")])
        finally:
            env.close()

        print(f"\nTASK {tid}: {name}")
        if desc:
            print(f"  lang: {desc}")
        print("  pos keys:", keys)


if __name__ == "__main__":
    main()
