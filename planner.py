import os
from pathlib import Path
import pddlgym
from pddlgym_planners.fd import FD
import shutil
import subprocess
import utils.utils as utils
import traceback

PLANNER = "./downward/fast-downward.py "
MAX_ERR_MSG_LEN = 10000000
PDDLGYM_PATH = os.path.dirname(pddlgym.__file__)

def SEARCH_CONFIG(mt): 
    return "--search 'astar(lmcut(), max_time={})' ".format(mt)

def query(domain_path: str, problem_path: str, plan_file: str, print_plan: False, max_time: float = 120):
    plan_file = os.path.join(os.getcwd(), plan_file)
    command = PLANNER + "--plan-file {} ".format(plan_file) +\
        domain_path + " " + problem_path + " " + SEARCH_CONFIG(max_time)
    print("Planning...")
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p.wait()
    exit_code, cost, plan_time, plan = 0, 0, 0., None
    if "Solution found" in str(output):
        for line in str(output).split("\\n"):
            if "Plan cost: " in line: cost = int(line.strip().split(" ")[-1])
            if "Planner time: " in line: plan_time = float(line.strip().split(" ")[-1].replace("s", ""))
        if os.path.isfile(plan_file):
            with open(plan_file, "r") as pf: plan = pf.read()
        exit_code = 1
    return plan, cost, plan_time, exit_code, err.decode()

def export_domain_to_pddlgym(domain: str, src_domain_file: str):
    dst = os.path.join(PDDLGYM_PATH, f"pddl/{domain}.pddl")
    if os.path.isfile(dst): os.remove(dst)
    shutil.copyfile(src_domain_file, dst)

def export_problem_to_pddlgym(domain: str, src_problem_file: str, p_idx: str, clear_dir=False):
    dst_path = os.path.join(PDDLGYM_PATH, f"pddl/{domain}/")
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    if clear_dir:
        for f in os.listdir(dst_path): os.remove(os.path.join(dst_path, f))
    shutil.copyfile(src_problem_file, os.path.join(dst_path, f"problem{p_idx}.pddl"))

def register_new_pddlgym_env(new_domain: str):
    new_env = (new_domain, {'operators_as_actions': True, 'dynamic_action_space': True})
    new_line = "\t\t" + str(new_env) + ",\n"
    with open(os.path.join(PDDLGYM_PATH, "__init__.py"), "r+") as file:
        lines = file.readlines()
        if new_line not in lines:
            for i, line in enumerate(lines):
                if "for env_name, kwargs in [" in line:
                    lines.insert(i + 1, new_line)
                    file.seek(0)
                    file.writelines(lines)
                    break

def query_pddlgym(domain: str, p_idx: int = 0, max_time: float = 120):
    plan, cost, node, time, exit_code = None, 0, 0, 0., 0
    print("Planning with undecomposed problem...")
    try:
        fd_planner = FD()
        fd_planner._alias = "seq-sat-lama-2011" # バグ回避のための属性直接指定
        env = pddlgym.make(f"PDDLEnv{domain.capitalize()}-v0")
        env.fix_problem_index(p_idx)
        state, _ = env.reset()
        plan = fd_planner(env.domain, state, timeout=max_time)
        stat = fd_planner.get_statistics()
        time, cost, node = stat["total_time"], stat["plan_cost"], stat["num_node_expansions"]
        print(f"Found solution in {time}s with cost {cost}")
        exit_code = 1
    except Exception:
        err = traceback.format_exc()
        print(f"Could not find solution!\n\n=== 🚨 ERROR DETAILS 🚨 ===\n{err}\n===========================\n")
        exit_code = 3 if "timed out" in err else 2
    return [p.pddl_str() for p in plan] if exit_code == 1 else None, time, node, cost, exit_code

def query_pddlgym_decompose(domain: str, subgoal_pddl_list: list, save_path: str = None, max_time: float = 120):
    digits = 2 if len(subgoal_pddl_list) >= 10 else 1
    p_0_file = os.path.join(PDDLGYM_PATH, f"pddl/{domain}/problem{str(0).zfill(digits)}.pddl")
    print("Planning with decomposed problems...")
    plans, times, nodes, costs, final_state, exit_code, completed_sp = [], [], [], [], [], 0, 0
    
    fd_planner = FD()
    fd_planner._alias = "seq-sat-lama-2011"

    for idx, sgp in enumerate(subgoal_pddl_list, start=1):
        sp_file = os.path.join(PDDLGYM_PATH, f"pddl/{domain}/problem{str(idx).zfill(digits)}.pddl")
        shutil.copyfile(p_0_file, sp_file)
        try:
            utils.set_pddl_problem_goal(sp_file, sgp)
            if idx > 1: utils.set_pddl_problem_init(sp_file, final_state)
        except Exception as e: print(f"Error writing PDDL: {e}")

        if save_path: shutil.copyfile(sp_file, os.path.join(save_path, f"p{str(idx).zfill(digits)}.pddl"))

        try:
            env = pddlgym.make(f"PDDLEnv{domain.capitalize()}-v0")
            env.fix_problem_index(idx)
            state, _ = env.reset()
            plan = fd_planner(env.domain, state, timeout=max_time)
            stat = fd_planner.get_statistics()
            print(f"Subgoal {idx}: {stat['total_time']}s")
            for act in plan: state, _, _, _, _ = env.step(act)
            final_state = sorted([lit.pddl_str() for lit in state.literals if not lit.is_negative])
            completed_sp += 1
            exit_code = 1
            plans.append([p.pddl_str() for p in plan])
            times.append(stat["total_time"]); nodes.append(stat["num_node_expansions"]); costs.append(stat["plan_cost"])
        except Exception:
            print(f"Subgoal {idx} failed!\n{traceback.format_exc()}")
            exit_code = 2; break
    return plans, times, nodes, costs, exit_code, completed_sp

def validate(domain_file: str, problem_file: str, plan_file: str):
    cmd = f"Validate -v {domain_file} {problem_file} {plan_file}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, _ = p.communicate()
    return ("Plan valid" in str(out)), str(out)

def val_feedback(err_msg: str):
    return "Plan failed: " + err_msg, 2