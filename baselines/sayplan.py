# Self-implemented code for the baseline approach SayPlan due to the lack of available open-source code
# Original paper: K. Rana et al., CoRL 2023, "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning"
# Paper link: https://proceedings.mlr.press/v229/rana23a/rana23a.pdf
# Implementation based on the prompt structure and input-output examples in the appendix of the paper


import argparse
from datetime import datetime
import pandas as pd
from pathlib import Path
import time
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data.scene_graph import load_scene_graph, prune_sg_with_item, collapsed_sg, update_sg, count_rooms
from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p
from utils.sayplan_utils import sayplan_plan_exp


# Parameters
# "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo", "gpt-4o"
DEFAULT_LLM = "gpt-4o"
TEMPERATURE = 0.
TOP_P = 1.
EPISODE = 5
MAX_DEBUG_ATTEMPTS = 4
MAX_SEARCH_ATTEMPTS = 50

# Examples
DOMAIN_EXAMPLE = "laundry"
SCENE_EXAMPLE = example.get_scenes(DOMAIN_EXAMPLE)[0]

# Queries
DOMAIN_QUERY = "pc"
SCENE_QUERY = example.get_scenes(DOMAIN_QUERY)[0]


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
    parser.add_argument("--domain-example",
                        action="store", type=str, dest="domain_example", default=DOMAIN_EXAMPLE,
                        choices=["home", "human", "laundry",
                                 "pc", "clean", "dining", "office"],
                        help="Choose domain example for prompting")
    parser.add_argument("-s", "--scene",
                        action="store", type=str, dest="scene", default=SCENE_QUERY,
                        choices=["allensville", "kemblesville", "pablo",
                                 "parole", "rosser", "shelbiana", "office"],
                        help="Choose scene graph for planning")
    parser.add_argument("--scene-example",
                        action="store", type=str, dest="scene_example", default=SCENE_EXAMPLE,
                        choices=["allensville", "kemblesville",
                                 "pablo", "parole", "rosser", "shelbiana", "office"],
                        help="Choose scene graph example for prompting")
    parser.add_argument("--max-debug-attempts",
                        action="store", type=int, dest="max_debug_attempts", default=MAX_DEBUG_ATTEMPTS,
                        help="Maximum number of debug attempts")
    parser.add_argument("--max-search-attempts",
                        action="store", type=int, dest="max_search_attempts", default=MAX_SEARCH_ATTEMPTS,
                        help="Maximum number of search attempts")
    parser.add_argument("--no-search",
                        action="store_true", dest="no_search", default=False,
                        help="No scene graph search")
    args = parser.parse_args()
    if args.scene == args.scene_example:
        raise argparse.ArgumentError(
            "Scene graph example cannot be identical to scene graph query!")
    print("Using model {}".format(args.model))
    print("Domain example: {} \nScene example: {} \nDomain query: {} \nScene query: {}".format(
        args.domain_example, args.scene_example, args.domain, args.scene))

    # Additional information for prompting
    exp = example.get_example(args.domain_example)
    qry = example.get_example(args.domain)
    if args.scene_example not in exp["scene"]:
        raise Exception("Scene example {} is not supported for domain example{}!".format(
            args.scene_example, args.domain_example))
    if args.scene not in qry["scene"]:
        raise Exception("Scene {} is not supported for domain {}!".format(
            args.scene, args.domain))
    add_obj_exp = exp["add_obj"]
    add_act_exp = exp["add_act"]
    add_state_exp = exp["env_state"]
    goal_exp = exp["goal"]
    items_keep_exp = exp["item_keep"]

    add_obj_qry = qry["add_obj"]
    add_act_qry = qry["add_act"]
    add_state_qry = qry["env_state"]
    goal_qry = qry["goal"]
    items_keep_qry = qry["item_keep"]
    gt_cost = qry["gt_cost"][args.scene]

    scene_exp = load_scene_graph(args.scene_example)
    scene_exp = prune_sg_with_item(scene_exp, items_keep_exp)
    scene_qry = load_scene_graph(args.scene)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S/")
    success = 0
    data_list = []

    # Loading LLM 
    model = llm.load_llm(args.model, args.temperature, args.top_p)

    for e in range(args.episode):
        model.reset()

        log_path = os.path.join(LOG_PATH(curr_time), "e_{:03}/".format(e))
        Path(log_path).mkdir(parents=True, exist_ok=True)
        csg_qry = collapsed_sg(scene_qry)
        # prompt_chain = []
        memory = set([])
        search_time, plan_time = 0., 0.
        plan_cost = 0
        replan_count = -1
        search_complete = False
        max_search_reached = False
        exit_code = 0

        ###################### Stage 1: Scene Graph Search ######################
        if not args.no_search:
            content_s, prompt_s = p.sayplan_search_prompt(
                goal_exp, goal_qry, csg_qry)
            search_count = 0
            print("Tokens for SayPlan Search Prompt {}: {}".format(
                search_count, model.count_tokens(prompt_s)))
            if args.print_prompt:
                model.log(content_s + prompt_s, os.path.join(
                    log_path, "sayplan_search_{}_{}_{}.prompt".format(args.domain, args.scene, search_count)))
            model.init_prompt_chain(content_s, prompt_s)
            # prompt_chain.extend([{"role": "system", "content": content_s},
            #                     {"role": "user", "content": prompt_s}])
            while not search_complete:
                start = time.time()
                output_tar = model.query_msg_chain()
                search_time += time.time() - start

                if args.print_response:
                    model.log(output_tar, os.path.join(
                        log_path, "sayplan_search_{}_{}_{}.response".format(args.domain, args.scene, search_count)))
                print("Response time for SayPlan ({}. search): {:.2f}s".format(
                    search_count, search_time))
                model.update_prompt_chain_w_response(output_tar)
                # prompt_chain.append(
                #     {"role": "assistant", "content": output_tar})

                # Export search command from response
                mode, cot, reasoning, cmd = llm_utils.export_sayplan_search_cmd(
                    output_tar)
                if "search complete" in cot or ("switch" in reasoning and "planning mode" in reasoning):
                    search_complete = True
                    print("Scene graph search complete! Switching to planning mode.")
                    break
                if len(memory) == count_rooms(scene_qry):
                    print("All rooms have been explored! Search stop!")
                    break
                if cmd[0] == "expand" and cmd[1] in memory:
                    prompt_s = "Room {} has already been explored! Remaining unexplored rooms are: {}".format(
                        cmd[1], set(scene_qry["rooms"].keys()) - memory)
                else:
                    csg_qry = update_sg(csg_qry, scene_qry, cmd[0], cmd[1])
                    memory.add(cmd[1])
                    prompt_s = "3D scene graph: {}\nMemory: {}".format(
                        csg_qry, memory)
                model.update_prompt_chain_w_response(prompt_s, role="user")
                # prompt_chain.append(
                #     {"role": "user", "content": prompt_s})

                search_count += 1
                if args.print_prompt:
                    model.log(prompt_s, os.path.join(
                        log_path, "sayplan_search_{}_{}_{}.prompt".format(args.domain, args.scene, search_count)))

                if search_count == args.max_search_attempts:
                    max_search_reached = True
                    exit_code = 6
                    print("Maximum search attempts reached!")
                    break

        ###################### Stage 2: Iterative Replanning ######################
        if not max_search_reached:
            if not args.no_search:
                content_p, prompt_p = p.sayplan_plan_prompt(
                    add_obj_exp, add_act_exp, add_state_exp,
                    add_obj_qry, add_act_qry, add_state_qry,
                    goal_exp, goal_qry, scene_exp, csg_qry)
                model.update_prompt_chain(content_p, prompt_p)
                # prompt_chain[0]["content"] = content_p
                # prompt_chain.append({"role": "user", "content": prompt_p})
            else:
                print("Skipping scene graph search.")
                scene_qry = prune_sg_with_item(scene_qry, items_keep_qry)
                output_exp = sayplan_plan_exp()
                output_exp.pop("mode")
                content_p, prompt_p = p.sayplan_prompt(add_obj_exp, add_act_exp, add_state_exp,
                                                       add_obj_qry, add_act_qry, add_state_qry,
                                                       goal_exp, scene_exp, output_exp,
                                                       goal_qry, scene_qry)
                model.update_prompt_chain(content_p, prompt_p)
                # prompt_chain.extend([{"role": "system", "content": content_p},
                #                     {"role": "user", "content": prompt_p}])

            print("Tokens for SayPlan Planning Prompt: {}".format(
                model.count_tokens(prompt_p)))
            if args.print_prompt:
                model.log(content_p + prompt_p, os.path.join(
                    log_path, "sayplan_plan_{}_{}.prompt".format(args.domain, args.scene)))

            for t in range(args.max_debug_attempts + 1):
                replan_count += 1

                start = time.time()
                output_tar = model.query_msg_chain()
                plan_time += time.time() - start

                if args.print_response:
                    model.log(output_tar, os.path.join(
                        log_path, "sayplan_replan_{}_{}_{}.response".format(args.domain, args.scene, t)))
                print("Response time for SayPlan ({}. attempt): {:.2f}s".format(
                    t, plan_time))
                model.update_prompt_chain_w_response(output_tar)

                plan_file = os.path.join(
                    log_path, "sayplan_{}_{}_{}.plan".format(args.domain, args.scene, t))
                try:
                    plan_list, plan_cost = llm_utils.export_sayplan_plan(
                        output_tar, plan_file)
                    print("Plan cost: {}, GT cost: {}".format(plan_cost, gt_cost))
                except Exception as err:
                    err_msg = "Error in exporting plan: {}".format(err)
                    model.update_prompt_chain_w_response(err_msg, role="user")
                    # prompt_chain.append({"role": "user", "content": err_msg})
                    print(err_msg)
                    exit_code = 7
                    continue

                is_valid, val_info = planner.validate(SRC_DOMAIN_PATH(
                    args.domain), SRC_PROBLEM_PATH(args.scene, args.domain), plan_file)
                if is_valid:
                    success += 1
                    exit_code = 1
                    break
                else:
                    # Replanning
                    feedback, exit_code = planner.val_feedback(val_info)
                    print("SayPlan Feedback: {}".format(feedback))
                    content_rp, prompt_rp = p.sayplan_replan_prompt(feedback)
                    print("Prompt tokens for replanning ({}. attempt): {}".format(
                        t+1, model.count_tokens(prompt_rp)))
                    if args.print_prompt:
                        model.log(content_rp + prompt_rp, os.path.join(log_path,
                                "sayplan_replan_{}_{}_{}.prompt".format(args.domain, args.scene, t+1)))
                    model.update_prompt_chain(content_rp, prompt_rp)
                    # prompt_chain[0]["content"] = content_rp
                    # prompt_chain.append({"role": "user", "content": prompt_rp})

        print("==================== Episode {}/{}, Total Success: {}, Replan: {} ====================".format(
            e + 1, args.episode, success, replan_count))
        data_list.append([e, exit_code, success, args.model, args.temperature,
                          args.domain_example, args.scene_example, args.domain, args.scene,
                          replan_count, search_time, plan_time, plan_cost, gt_cost])

    df = pd.DataFrame(data_list, columns=["Episode", "Exit Code", "Success", "LLM", "Temp",
                                          "Domain Exp", "Scene Exp", "Domain Qry", "Scene Qry",
                                          "Replan Count", "Search Time", "Plan Time", "Plan Cost", "GT Cost"])
    df.to_csv(os.path.join(LOG_PATH(curr_time), "log.csv"))
    print("Success rate: {:.2f}%".format(success / args.episode * 100.))
