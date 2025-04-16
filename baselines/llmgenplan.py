# Re-implemented code for the baseline approach LLM-GenPlan
# Original paper: T. Silver et al., AAAI 2024, "Generalized Planning in PDDL Domains with Pretrained Large Language Models"
# Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/30006
# Source code with MIT License: https://github.com/tomsilver/llm-genplan


import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
import signal
import time
import traceback
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p
from utils.llmgenplan_utils import PDDLTask, GeneralizedPlan


# Parameters
# "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo", "gpt-4o"
DEFAULT_LLM = "gpt-4o"
TEMPERATURE = 0.
TOP_P = 1.
EPISODE = 5
MAX_TIME = 60
MAX_DEBUG_ATTEMPTS = 4

# Queries and examples
DOMAIN_QUERY = "pc"
SCENE_QUERY = example.get_scenes(DOMAIN_QUERY)[0]


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!")


def SRC_DOMAIN_PATH(d):
    return "data/pddl/domain/{}_domain.pddl".format(d)


def SRC_PROBLEM_PATH(s, d):
    return "data/pddl/problem/{}_{}_problem.pddl".format(s, d)


def LOG_PATH(t):
    return "result/{}".format(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model",
                        action="store", type=str, dest="model", default=DEFAULT_LLM,
                        choices=["gpt-35-turbo-16k", "gpt-4",
                                 "gpt-4-turbo", "gpt-4o",
                                 "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct"],
                        help="Choose Large Language Model")
    parser.add_argument("-t", "--temperature",
                        action="store", type=float, dest="temperature", default=TEMPERATURE,
                        help="Temperature parameter for the LLM")
    parser.add_argument("--top-p",
                        action="store", type=float, dest="top_p", default=TOP_P,
                        help="Top p parameter for the LLM")
    parser.add_argument("-e", "--episode",
                        action="store", type=int, dest="episode", default=EPISODE,
                        help="Number of epsiodes of the experiment")
    parser.add_argument("-p", "--print-prompt",
                        action="store_true", dest="print_prompt", default=True,
                        help="Print prompt")
    parser.add_argument("-r", "--print-response",
                        action="store_true", dest="print_response", default=True,
                        help="Print response")
    parser.add_argument("--print-plan",
                        action="store_true", dest="print_plan", default=False,
                        help="Print plan")
    parser.add_argument("-d", "--domain",
                        action="store", type=str, dest="domain", default=DOMAIN_QUERY,
                        choices=["home", "human", "laundry",
                                 "pc", "clean", "dining", "office"],
                        help="Choose domain for querying")
    parser.add_argument("-s", "--scene",
                        action="store", type=str, dest="scene", default=SCENE_QUERY,
                        choices=["allensville", "kemblesville", "pablo",
                                 "parole", "rosser", "shelbiana", "office"],
                        help="Choose scene for querying")
    parser.add_argument("--max-time",
                        action="store", type=int, dest="max_time", default=MAX_TIME,
                        help="Time limit for task planner [s]")
    parser.add_argument("--max-debug-attempts",
                        action="store", type=int, dest="max_debug_attempts", default=MAX_DEBUG_ATTEMPTS,
                        help="Maximum number of debug attempts")
    args = parser.parse_args()

    # Get scene examples
    scene_exps = [sc for sc in example.get_scenes(
        args.domain) if sc != args.scene]
    print("Using model {}".format(args.model))
    print("Domain query: {} \nScene query: {} \nScene example(s): {}".format(
        args.domain, args.scene, scene_exps))

    with open(SRC_DOMAIN_PATH(args.domain), "r") as df:
        domain_qry = df.read()

    problem_exp_list = []
    for sce in scene_exps:
        with open(SRC_PROBLEM_PATH(sce, args.domain), "r") as pxf:
            problem_exp = pxf.read()
        problem_exp_list.append(problem_exp)

    with open(SRC_PROBLEM_PATH(args.scene, args.domain), "r") as pf:
        problem_qry = pf.read()

    gt_cost = example.get_example(args.domain)["gt_cost"][args.scene]
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S/")
    success = 0
    data_list = []
    
    # Loading LLM
    model = llm.load_llm(args.model, args.temperature, args.top_p)

    for e in range(args.episode):
        model.reset()
        
        log_path = os.path.join(LOG_PATH(curr_time), "e_{:03}/".format(e))
        Path(log_path).mkdir(parents=True, exist_ok=True)

        ###################### Stage 1: Domain Summary ######################
        content_ds, prompt_ds = p.llmgenplan_domain_summary(
            domain_qry, problem_exp_list)
        print("Prompt tokens for domain summary: {}".format(
            model.count_tokens(prompt_ds)))
        if args.print_prompt:
            model.log(content_ds + prompt_ds, os.path.join(
                    log_path, "{}_{}_domain_sum.prompt".format(args.scene, args.domain)))
        model.init_prompt_chain(content_ds, prompt_ds)

        ds_start = time.time()
        domain_summary = model.query_msg_chain()
        ds_time = time.time() - ds_start

        if args.print_response:
            model.log(domain_summary, os.path.join(
                log_path, "{}_{}_domain_sum.response".format(args.scene, args.domain)))
        print("Response time for domain summary: {:.2f}s".format(ds_time))
        model.update_prompt_chain_w_response(domain_summary)

        ###################### Stage 2: Strategy Proposal ######################
        content_sp, prompt_sp = p.llmgenplan_strategy()
        print("Prompt tokens for strategy proposal: {}".format(
            model.count_tokens(prompt_sp)))
        if args.print_prompt:
            model.log(content_sp + prompt_sp, os.path.join(
                    log_path, "{}_{}_strategy.prompt".format(args.scene, args.domain)))
        model.update_prompt_chain(content_sp, prompt_sp)

        sp_start = time.time()
        strategy_prop = model.query_msg_chain()
        sp_time = time.time() - sp_start

        if args.print_response:
            model.log(strategy_prop, os.path.join(
                log_path, "{}_{}_strategy.response".format(args.scene, args.domain)))
        print("Response time for strategy proposal: {:.2f}s".format(sp_time))
        model.update_prompt_chain_w_response(strategy_prop)

        ###################### Stage 3: Python Implementation ######################
        task = PDDLTask(domain_qry, problem_qry)
        last_err_info, last_err_type = None, None
        plan = []
        plan_time = 0
        replan_count = -1
        results = []
        for t in range(args.max_debug_attempts + 1):
            replan_count += 1
            if t == 0:
                content_pi, prompt_pi = p.llmgenplan_impl_func(
                    typed=task.typed)
                print("Prompt tokens for Python implementation ({}. attempt): {}".format(
                    t, model.count_tokens(prompt_pi)))
                if args.print_prompt:
                    model.log(content_pi + prompt_pi, os.path.join(log_path,
                            "{}_{}_python_replan_{}.prompt".format(args.scene, args.domain, t)))
                model.update_prompt_chain(content_pi, prompt_pi)
            else:
                assert last_err_info is not None and last_err_type is not None
                content_rp, prompt_rp = p.llmgenplan_replan(last_err_info)
                print("Prompt tokens for Python implementation ({}. attempt): {}".format(
                    t, model.count_tokens(prompt_rp)))
                if args.print_prompt:
                    model.log(content_rp + prompt_rp, os.path.join(log_path,
                            "{}_{}_python_replan_{}.prompt".format(args.scene, args.domain, t)))
                model.update_prompt_chain(content_rp, prompt_rp)

            # Generate plan with Python code
            pi_start = time.time()
            python_impl = model.query_msg_chain()
            pi_time = time.time() - pi_start
            plan_time += pi_time

            if args.print_response:
                model.log(python_impl, os.path.join(
                    log_path, "{}_{}_python_replan_{}.response".format(args.scene, args.domain, t)))
            print("Response time for Python implementation ({}. attempt): {:.2f}s".format(
                t, pi_time))
            model.update_prompt_chain_w_response(python_impl)

            # Parse Python code from response
            code_str = llm_utils.export_python_code(python_impl)
            gen_plan_path = os.path.join(log_path, "gen_plan_{}.py".format(t))
            gen_plan_code = GeneralizedPlan(code_str, gen_plan_path)

            # Execute Python code and validate plan
            # Register the signal function handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(args.max_time)
            try:
                plan = gen_plan_code.run(task)
            except (TimeoutException, BaseException):
                tb = traceback.format_exc()
                if "Timed out!" in tb:
                    last_err_info = "The code was interrupted because it timed out (possible infinite loop)."
                    last_err_type = "python-timeout"
                    results.append({"success": False,
                                    "err-info": last_err_info, "err-type": last_err_type})
                else:
                    msg = f"The code raised the following exception:\n{tb}"
                    # last_err_info = f"Given this task:\n{task.problem_str}\n{msg}"
                    last_err_info = msg
                    last_err_type = "python-exception"
                    results.append({"success": False,
                                    "err-info": last_err_info, "err-type": last_err_type})
                print("Exception: {}".format(last_err_type))
                continue
            finally:
                # Cancel the alarm
                signal.alarm(0)

            if not isinstance(plan, list):
                msg = f"The code returned {plan}, which is not a list of actions."
                # last_err_info = f"Given this task:\n{task.problem_str}\n{msg}"
                last_err_info = msg
                last_err_type = "output-not-plan"
                results.append({"success": False,
                                "err-info": last_err_info, "err-type": last_err_type})
                print("Error: {}".format(last_err_type))
                continue
            if any(not isinstance(elem, str) for elem in plan):
                msg = f"The code returned {plan}, which contains non-string actions."
                # last_err_info = f"Given this task:\n{task.problem_str}\n{msg}"
                last_err_info = msg
                last_err_type = "action-not-str"
                results.append({"success": False,
                                "err-info": last_err_info, "err-type": last_err_type})
                print("Error: {}".format(last_err_type))
                continue
            plan_file = os.path.join(
                log_path, "{}_{}_{}.plan".format(args.domain, args.scene, t))
            with open(plan_file, "w") as pf:
                pf.write("\n".join(plan))

            # Check syntax.
            syntax_err_found = False
            for i, action in enumerate(plan):
                if not task.action_has_valid_syntax(action):
                    msg = (
                        f"The code returned this plan: {plan}\n"
                        f"However, the action {action} is invalid at step {i}.\n"
                        f"NOTE: the valid operators are: {task.actions_hint}."
                    )
                    # last_err_info = f"Given this task:\n{task.problem_str}\n{msg}"
                    last_err_info = msg
                    last_err_type = "operator-syntax-invalid"
                    syntax_err_found = True
                    results.append({"success": False,
                                    "err-info": last_err_info, "err-type": last_err_type})
                    break
            if syntax_err_found:
                print("Error: {}".format(last_err_type))
                continue

            # Check semantics
            is_valid, val_info = planner.validate(
                SRC_DOMAIN_PATH(args.domain), SRC_PROBLEM_PATH(args.scene, args.domain), plan_file)
            # if is_valid or len(plan) == gt_cost:
            if is_valid:
                success += 1
                last_err_info = "Generalized plan succeeded."
                results.append({"success": True,
                                "err-info": last_err_info, "err-type": None})
                print(last_err_info)
                break

            msg = f"The code failed. It returned the following plan: {plan}.\n{val_info}"
            # last_err_info = f"Given this task:\n{task.problem_str}\n{msg}"
            last_err_info = msg
            last_err_type = "operator-semantics-invalid"
            results.append({"success": False,
                            "err-info": last_err_info, "err-type": last_err_type})
            # print("Error: {}".format(last_err_type))

        print("==================== Episode {}/{}, Total Success: {}, Replan: {} ====================".format(
            e + 1, args.episode, success, replan_count))
        data_list.append([e, success, args.model, args.temperature,
                          args.domain, args.scene, scene_exps,
                          ds_time, sp_time, plan_time, replan_count, len(plan), gt_cost, results])

    df = pd.DataFrame(data_list, columns=["Episode", "Success", "LLM", "Temp",
                                          "Domain Qry", "Scene Qry", "Scene Exp(s)",
                                          "Time Domain Sum", "Time Strategy", "Time Python Impl",
                                          "Replan Count", "Plan Cost", "GT Cost", "Results"])
    df.to_csv(os.path.join(LOG_PATH(curr_time), "log.csv"))
    print("Success rate: {:.2f}%".format(success / args.episode * 100.))
