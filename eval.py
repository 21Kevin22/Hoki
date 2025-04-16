import copy
import csv
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def read_csv_column(file_path, column_name):
    column_data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            column_data.append(row[column_name])
    return column_data


def read_csv_row_succeed(file_path):
    plan_t, plan_t_decomp = [], []
    node_e, node_e_decomp = [], []
    cost, cost_decomp = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i, acc_s_orig, acc_s_decomp = 0, 0, 0
        for row in reader:
            if i > 0:
                # Check success original problem
                if eval(row[3]) - acc_s_orig == 1:
                    plan_t.append(eval(row[19]))
                    node_e.append(eval(row[21]))
                    cost.append(eval(row[20]))
                acc_s_orig = eval(row[3])
                # Check success decomposed problem
                if eval(row[2]) - acc_s_decomp == 1:
                    plan_t_decomp.append(eval(row[23]))
                    node_e_decomp.append(eval(row[25]))
                    cost_decomp.append(eval(row[27]))
                acc_s_decomp = eval(row[2])
            i += 1
    return plan_t, plan_t_decomp, node_e, node_e_decomp, cost, cost_decomp


def print_results(file_path):
    plan_t, plan_t_decomp, node_e, node_e_decomp, cost, cost_decomp = read_csv_row_succeed(
        file_path)
    print("Len plan time: {}, decomp: {}".format(
        len(plan_t), len(plan_t_decomp)))
    print("Len node expanded: {}, decomp: {}".format(
        len(node_e), len(node_e_decomp)))
    if len(plan_t) > 0:
        print("Cost mean: {}, std: {}".format(
            np.mean(cost), np.std(cost)))
        print("Plan time mean: {:.2f}, std: {:.4f}".format(
            np.mean(plan_t), np.std(plan_t)))
        print("Node expanded mean: {}, std: {}".format(
            np.round(np.mean(node_e)), np.round(np.std(node_e))))
    if len(plan_t_decomp) > 0:
        print("Cost decomp mean: {}, std: {}".format(
            np.mean(cost_decomp), np.std(cost_decomp)))
        print("Plan time decomp mean: {:.4f}, std: {:.4f}".format(
            np.mean(plan_t_decomp), np.std(plan_t_decomp)))
        print("Node expanded decomp mean: {}, std: {}".format(
            round(np.mean(node_e_decomp)), round(np.std(node_e_decomp))))

    column_names = ['Time Domain', 'Time Prune',
                    'Time Problem', 'Time Decomp', 'Total LLM Time']
    for column_name in column_names:
        column_data = read_csv_column(file_path, column_name)
        print("{}: mean: {:.2f}, std: {:.2f}".format(column_name,
              np.mean([eval(elem) for elem in column_data]),
              np.std([eval(elem) for elem in column_data])))


def print_results_delta(file_path):
    domain_time, prune_time, prob_time, decomp_time, total_llm_time = [], [], [], [], []
    plan_time, plan_time_decomp, node_exp, node_exp_decomp, cost, cost_decomp = \
        [], [], [], [], [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i > 0:
                if eval(row[2]) == 1:
                    domain_time.append(eval(row[13]))
                    prune_time.append(eval(row[14]))
                    prob_time.append(eval(row[15]))
                    decomp_time.append(eval(row[16]))
                    total_llm_time.append(eval(row[17]))
                    plan_time.append(eval(row[20]))
                    cost.append(eval(row[21]))
                    node_exp.append(eval(row[22]))
                if eval(row[3]) == 1:
                    plan_time_decomp.append(eval(row[23]))
                    cost_decomp.append(eval(row[27]))
                    node_exp_decomp.append(eval(row[25]))
            i += 1
    print("Len llm time: {}, plan cost: {}".format(
        len(domain_time), len(cost)))
    if len(cost) > 0:
        print("Domain time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in domain_time]), np.std([elem for elem in domain_time])))
        print("Prune time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in prune_time]), np.std([elem for elem in prune_time])))
        print("Problem time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in prob_time]), np.std([elem for elem in prob_time])))
        print("Decomp time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in decomp_time]), np.std([elem for elem in decomp_time])))
        print("Total LLM time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in total_llm_time]), np.std([elem for elem in total_llm_time])))
        print("Cost mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in cost]), np.std([elem for elem in cost])))
        print("Plan time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in plan_time]), np.std([elem for elem in plan_time])))
        print("Node expanded mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in node_exp]), np.std([elem for elem in node_exp])))
    if len(cost_decomp) > 0:
        print("Cost decomp mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in cost_decomp]), np.std([elem for elem in cost_decomp])))
        print("Plan time decomp mean: {:.4f}, std: {:.4f}".format(np.mean(
            [elem for elem in plan_time_decomp]), np.std([elem for elem in plan_time_decomp])))
        print("Node expanded decomp mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in node_exp_decomp]), np.std([elem for elem in node_exp_decomp])))


def print_results_sayplan(file_path, no_replan=False):
    replan_count, search_time, plan_time, plan_cost = [], [], [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i > 0 and eval(row[2]) == 1:
                if no_replan:
                    if eval(row[-5]) == 0:
                        replan_count.append(eval(row[10]))
                        search_time.append(eval(row[11]))
                        plan_time.append(eval(row[12]))
                        plan_cost.append(eval(row[13]))
                else:
                    replan_count.append(eval(row[10]))
                    search_time.append(eval(row[11]))
                    plan_time.append(eval(row[12]))
                    plan_cost.append(eval(row[13]))
            i += 1
    print("Len llm time: {}, plan cost: {}".format(
        len(search_time), len(plan_cost)))
    if len(plan_cost) > 0:
        print("Replan count mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in replan_count]), np.std([elem for elem in replan_count])))
        print("Search time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in search_time]), np.std([elem for elem in search_time])))
        print("Plan time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in plan_time]), np.std([elem for elem in plan_time])))
        print("Plan cost mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in plan_cost]), np.std([elem for elem in plan_cost])))


def print_results_llmgenplan(file_path, no_replan=False):
    domain_time, stg_time, py_time, replan_count, plan_cost = [], [], [], [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i > 0:
                if no_replan:
                    if eval(row[-1])[0]["success"]:
                        domain_time.append(eval(row[8]))
                        stg_time.append(eval(row[9]))
                        py_time.append(eval(row[10]))
                        replan_count.append(eval(row[11]))
                        plan_cost.append(eval(row[12]))
                else:
                    if eval(row[-1])[-1]["success"]:
                        domain_time.append(eval(row[8]))
                        stg_time.append(eval(row[9]))
                        py_time.append(eval(row[10]))
                        replan_count.append(eval(row[11]))
                        plan_cost.append(eval(row[12]))
            i += 1
    print("Len llm time: {}, plan cost: {}".format(
        len(domain_time), len(plan_cost)))
    if len(plan_cost) > 0:
        print("Replan count mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in replan_count]), np.std([elem for elem in replan_count])))
        print("Domain time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in domain_time]), np.std([elem for elem in domain_time])))
        print("Strategy time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in stg_time]), np.std([elem for elem in stg_time])))
        print("Python time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in py_time]), np.std([elem for elem in py_time])))
        print("Plan cost mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in plan_cost]), np.std([elem for elem in plan_cost])))


def print_results_llmasplanner(file_path):
    llm_time, plan_cost = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i > 0:
                if eval(row[2]) == 1:
                    llm_time.append(eval(row[-3]))
                    plan_cost.append(eval(row[-2]))
            i += 1
    print("Len llm time: {}, plan cost: {}".format(
        len(llm_time), len(plan_cost)))
    if len(plan_cost) > 0:
        print("LLM time mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in llm_time]), np.std([elem for elem in llm_time])))
        print("Plan cost mean: {:.2f}, std: {:.4f}".format(np.mean(
            [elem for elem in plan_cost]), np.std([elem for elem in plan_cost])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process log files for evaluation.")
    parser.add_argument("-f", "--file",
                        action="store", type=str, dest="file_path", required=True,
                        help="Path to the log file")
    args = parser.parse_args()
    file_path = args.file_path

    print_results(file_path)
    # print_results_delta(file_path)
    # print_results_sayplan(file_path, no_replan=True)
    # print_results_llmgenplan(file_path)
    # print_results_llmgenplan(file_path, no_replan=True)
    # print_results_llmasplanner(file_path)
