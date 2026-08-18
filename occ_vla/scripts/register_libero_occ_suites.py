"""
register_libero_occ_suites.py

Registers the 4 real LIBERO-Occ (arXiv:2606.10862, github.com/litsh/Libero-Occ)
occluded task suites (libero_{spatial,object,goal,10}_occluded) with
libero.libero.benchmark, WITHOUT editing the vendored LIBERO source
(thirdparty/LIBERO/libero/libero/benchmark/{__init__,libero_suite_task_map}.py).
Purely additive at runtime: extends the module-level `libero_task_map` dict
and calls the existing `register_benchmark` decorator with new Benchmark
subclasses, mirroring exactly what LIBERO_10/LIBERO_GOAL/etc. already do for
the stock suites.

Unlike our own `apply_partial_patch` (a synthetic, fixed, centered gray
2D-image patch), this benchmark's occlusion is REAL SCENE GEOMETRY: the
"_occluded" BDDL variants add real 3D objects (e.g. wooden_cabinet_1,
short_cabinet_1) into the scene at fixed positions that physically block
camera line-of-sight to the target -- confirmed by diffing
KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl (libero_10) against its
libero_10_occluded counterpart (occ_vla 2026-08-09). This is a fundamentally
different, much more standard/externally-validated occlusion mechanism than
our synthetic patch, and lets us report numbers against a real published
benchmark.

Prerequisite: scripts/setup/install_libero_occ_assets.sh (in
third_party/Libero-Occ) must already have been run against this LIBERO
checkout (copies bddl_files/init_files into thirdparty/LIBERO/libero/libero/).
This module only does the Python-level benchmark registration, not the file
copy.

Usage: `import register_libero_occ_suites` (or run this file directly for a
quick smoke test) BEFORE calling `benchmark.get_benchmark_dict()` /
`benchmark.get_benchmark(name)` -- e.g. before `run_libero_eval.py`'s own
benchmark lookup. Idempotent (re-importing is a no-op after the first call).
"""

import glob
import os

from libero.libero import get_libero_path
from libero.libero.benchmark import (
    Benchmark,
    Task,
    grab_language_from_filename,
    register_benchmark,
    task_maps,
)
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

OCCLUDED_SUITES = [
    "libero_spatial_occluded",
    "libero_object_occluded",
    "libero_goal_occluded",
    "libero_10_occluded",
]

_already_registered = False


def _discover_task_names(suite_name):
    bddl_dir = os.path.join(get_libero_path("bddl_files"), suite_name)
    assert os.path.isdir(bddl_dir), (
        f"{bddl_dir} not found -- run "
        f"`LIBERO_ROOT=<this LIBERO checkout> bash "
        f"third_party/Libero-Occ/scripts/setup/install_libero_occ_assets.sh` first"
    )
    names = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(bddl_dir, "*.bddl"))
    )
    assert names, f"no .bddl files found in {bddl_dir}"
    return names


def register_all():
    global _already_registered
    if _already_registered:
        return
    for suite_name in OCCLUDED_SUITES:
        task_names = _discover_task_names(suite_name)
        libero_task_map[suite_name] = task_names  # extends the vendored dict in-memory only

        task_maps[suite_name] = {}
        for task in task_names:
            language = grab_language_from_filename(task + ".bddl")
            task_maps[suite_name][task] = Task(
                name=task,
                language=language,
                problem="Libero",
                problem_folder=suite_name,
                bddl_file=f"{task}.bddl",
                init_states_file=f"{task}.pruned_init",
            )

        # register_benchmark keys off target_class.__name__.lower() -- must
        # preserve underscores (bug caught here: an earlier CamelCase attempt
        # collapsed them, producing "liberospatialoccluded" != the intended
        # "libero_spatial_occluded" lookup key) to match get_benchmark(name)'s
        # exact lookup string, mirroring the vendored LIBERO_SPATIAL etc. convention.
        class_name = suite_name.upper()

        def _make_init(_suite_name):
            def __init__(self, task_order_index=0):
                Benchmark.__init__(self, task_order_index=task_order_index)
                self.name = _suite_name
                # occluded suites keep BDDL-file order (task_orders indices only
                # cover the stock 10-task permutations, not these) -- same
                # simplification LIBERO_90 already uses.
                self.tasks = list(task_maps[self.name].values())
                self.n_tasks = len(self.tasks)

            return __init__

        new_cls = type(class_name, (Benchmark,), {"__init__": _make_init(suite_name)})
        register_benchmark(new_cls)
        print(f"[register_libero_occ_suites] registered '{suite_name}' -> {class_name} "
              f"({len(task_names)} tasks)")

    _already_registered = True


register_all()


if __name__ == "__main__":
    from libero.libero.benchmark import get_benchmark, get_benchmark_dict

    print("\nAll registered benchmarks:", sorted(get_benchmark_dict().keys()))
    for suite_name in OCCLUDED_SUITES:
        b = get_benchmark(suite_name)()
        print(f"\n{suite_name}: {b.get_num_tasks()} tasks")
        for i, name in enumerate(b.get_task_names()):
            print(f"  [{i}] {name}")
