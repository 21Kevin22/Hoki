"""Candidate tasks of PREREG_main.md §2 (all in pi0.5-LIBERO's 40 training tasks; checked against tasks.jsonl)."""
TASKS = {
    "T1": ("libero_10", 9, "put the yellow and white mug in the microwave and close it"),
    "T2": ("libero_10", 3, "put the black bowl in the bottom drawer of the cabinet and close it"),
    "T3": ("libero_goal", 3, "open the top drawer and put the bowl inside"),
    "T4": ("libero_goal", 0, "open the middle drawer of the cabinet"),
    "T5": ("libero_10", 5, "pick up the book and place it in the back compartment of the caddy"),
    "T6": ("libero_spatial", 4, "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"),
    "T7": ("libero_10", 0, "put both the alphabet soup and the tomato sauce in the basket"),
    # PREREG 2026-11-23 (2): the rest of libero_10, for the failure-cause study (not candidates of §2)
    "T8": ("libero_10", 1, "put both the cream cheese box and the butter in the basket"),
    "T9": ("libero_10", 2, "turn on the stove and put the moka pot on it"),
    "T10": ("libero_10", 4, "put the white mug on the left plate and put the yellow and white mug on the right plate"),
    "T11": ("libero_10", 6, "put the white mug on the plate and put the chocolate pudding to the right of the plate"),
    "T12": ("libero_10", 7, "put both the alphabet soup and the cream cheese box in the basket"),
    "T13": ("libero_10", 8, "put both moka pots on the stove"),
}
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)   # openpi main.py


def bddl_path(suite, idx):
    import os
    from libero.libero import benchmark, get_libero_path
    t = benchmark.get_benchmark_dict()[suite]().get_task(idx)
    return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), t.language
