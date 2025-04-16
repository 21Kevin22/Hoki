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

from data.scene_graph import load_scene_graph, prune_sg_with_item
from data import example
import llm.llm as llm
from llm import llm_utils
import planner
import prompt as p


# Parameters
# "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo", "gpt-4o"
DEFAULT_LLM = "gpt-4o"
TEMPERATURE = 0.
TOP_P = 1.
EPISODE = 5

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
    goal_exp = exp["goal"]
    items_keep_exp = exp["item_keep"]

    add_obj_qry = qry["add_obj"]
    add_act_qry = qry["add_act"]
    goal_qry = qry["goal"]
    items_keep_qry = qry["item_keep"]
    gt_cost = qry["gt_cost"][args.scene]

    scene_exp = load_scene_graph(args.scene_example)
    scene_exp = prune_sg_with_item(scene_exp, items_keep_exp)
    scene_qry = load_scene_graph(args.scene)
    scene_qry = prune_sg_with_item(scene_qry, items_keep_qry)

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S/")
    success = 0
    data_list = []
    
    # Loading LLM
    model = llm.load_llm(args.model, args.temperature, args.top_p)

    for e in range(args.episode):
        model.reset()

        log_path = os.path.join(LOG_PATH(curr_time), "e_{:03}/".format(e))
        Path(log_path).mkdir(parents=True, exist_ok=True)
        plan_cost = 0
        exit_code = 0

        content, prompt = p.sg_2_plan(scene_exp, scene_qry, goal_exp, goal_qry,
                                      add_obj_exp, add_obj_qry, add_act_exp, add_act_qry)
        print("Tokens for LLM-As-Planner Prompt: {}".format(
            model.count_tokens(prompt)))
        if args.print_prompt:
            model.log(content + prompt, os.path.join(
                log_path, "llmasplanner_{}_{}.prompt".format(args.domain, args.scene)))

        # Query LLM
        start = time.time()
        output_tar = model.query(content, prompt)
        llm_time = time.time() - start

        if args.print_response:
            model.log(output_tar, os.path.join(
                log_path, "llmasplanner_{}_{}.response".format(args.domain, args.scene)))
        print("Response time for LLM-As-Planner: {:.2f}s".format(llm_time))

        # Export plan
        plan_file = os.path.join(
            log_path, "llmasplanner_{}_{}.plan".format(args.domain, args.scene))
        try:
            plan_list, plan_cost = llm_utils.export_sayplan_plan(
                output_tar, plan_file)
            print("Plan cost: {}, GT cost: {}".format(plan_cost, gt_cost))
        except Exception as err:
            err_msg = "Error in exporting plan: {}".format(err)
            print(err_msg)
            # exit_code = 0

        # Validate plan
        is_valid, val_info = planner.validate(SRC_DOMAIN_PATH(
            args.domain), SRC_PROBLEM_PATH(args.scene, args.domain), plan_file)
        if is_valid:
            success += 1
            exit_code = 1
        else:
            feedback, exit_code = planner.val_feedback(val_info)
            print("Feedback: {}".format(feedback))

        print("==================== Episode {}/{}, Total Success: {} ====================".format(
            e + 1, args.episode, success))
        data_list.append([e, exit_code, success, args.model, args.temperature,
                          args.domain_example, args.scene_example, args.domain, args.scene,
                          llm_time, plan_cost, gt_cost])

    df = pd.DataFrame(data_list, columns=["Episode", "Exit Code", "Success", "LLM", "Temp",
                                          "Domain Exp", "Scene Exp", "Domain Qry", "Scene Qry",
                                          "LLM Time", "Plan Cost", "GT Cost"])
    df.to_csv(os.path.join(LOG_PATH(curr_time), "log.csv"))
    print("Success rate: {:.2f}%".format(success / args.episode * 100.))
