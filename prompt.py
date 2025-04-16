import itertools
import os
import utils.utils as utils
from utils.sayplan_utils import sayplan_output_format, sayplan_search_exp, sayplan_plan_exp


actions = """
    For example, a domain has the following object types: agent, room, and item. The agent can perform the following basic actions:
    goto(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>, where <room_1> and <room_2> should be neighbors. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be accessible and located in <room>, the pick action is in <item>'s affordance, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <item> is accessible, the drop action is in <item>'s affordance, <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    turnon(<agent>, <item>, <room>): <agent> turns on an <item> at <room>. <item> is accessible, the turnon action is in <item>'s affordance, both <agent> and <item> are in <room>, <agent> isn't loaded, and <item> state is off. As result, <item> state will change to on.
    turnoff(<agent>, <item>, <room>): <agent> turns off an <item> at <room>. <item> is accessible, the turnoff action is in <item>'s affordance, both <agent> and <item> are in <room>, <agent> isn't loaded, and <item> state is on. As result, <item> state will change to off.
    open(<agent>, <item>, <room>): <agent> opens an <item> in <room>. <item> is accessible, the open action is in <item>'s affordance, both <agent> and <item> are in <room>, <agent> isn't loaded, and <item> state is closed. As result, <item> state will change to open.
    close(<agent>, <item>, <room>): <agent> closes an <item> in <room>. <item> is accessible, the close action is in the <item>'s affordance, both <agent> and <item> are in <room>, <agent> isn't loaded,and <item> state is open. As result, <item> state will change to closed.
"""

sg_example = {
    "bathroom": {
        "items": {
            "toilet": {
                "accessible": False,
                "affordance": [],
                "state": "free"
            },
            "detergent": {
                "accessible": True,
                "affordance": ["pick", "drop"],
                "state": "free"
            }
        },
        "neighbor": ["corridor_1", "closet_1"]
    }
}

sg_exp_str = """
    E.g., a bathroom in a scene graph is defined as:
    ```\n{}\n```
    There are three items in the bathroom, so the postions can be defined in PDDL using the predicate "item_at" defined in the domain file as follows:
    ```
    (item_at toilet bathroom)
    (item_at detergent bathroom)
    ```
    Among them, the "accessible" attribute of detergent is "True", which results in a "item_accessible" predicate (previously defined in the domain file) as follows:
    ```(item_accessible detergent)```
    and the "affordance" attribute has "pick" and "drop", which results in a "item_pickable" predicate (previously defined in the domain file) as follows:
    ```(item_pickable detergent)```
    
    Don't forget the "is" predicates in the domain file, e.g. if there is a predicate ```(item_is_detergent ?i - item)``` defined in the domain file,
    and the scene graph also contains a "detergent" item, then the problem file should also include the predicate ```(item_is_detergent detergent)``` in the ; Attributes part of the "(:init )" section.
""".format(sg_example)


def p_template(x): return """
    Here is an example of a scene graph in the form of a nested dictionary in Python: \n{}\n
    The top level keys are the name of the scene, the rooms, the agent current position and state (and possibly also the human positions states). Each room contains a dictionary of 'objects' inside the rooms, and a list of 'neighbor' rooms of the current room.
    Each object has three attributes, 'accessible' means if the object can be accessed or not, 'affordance' indicates the affordable actions of this object, 'state' infers whether the object is free, or occupied, e.g. being picked by an agent.
    
    Here are the basic actions the agent can do:
    goto(<room>): Move the agent to <room>, which must be the neighbor room of the agent's current position. As result, the agent position after the action will be <room>.
    pick(<item>): Pick up an <item> if the <item> is accessible, the pick action is in the <item>'s affordance, the <item> state is free, and the agent state is hand-free. As result, the <item> state will change to in-hand, the agent state will change to loaded.
    drop(<item>): Drop an <item> if the <item> is accessible, the drop action is in the <item>'s affordance, the <item> state is in-hand, and the agent state is loaded. As result, the <item> state will change to free, the agent state will change to hand-free.
    turnon(<item>): Turn on an <item> if the <item> is accessible, the turnon action is in the <item>'s affordance, the <item> state is off, and the agent state is hand-free. As result, the <item> state will change to on.
    turnoff(<item>): Turn off an <item> if the <item> is accessible, the turnoff action is in the <item>'s affordance, the <item> state is on, and the agent state is hand-free. As result, the <item> state will change to off.
    open(<item>): Open an <item> if the <item> is accessible, the open action is in the <item>'s affordance, the <item> state is closed, and the agent state is hand-free. As result, the <item> state will change to open.
    close(<item>): Close an <item> if the <item> is accessible, the close action is in the <item>'s affordance, the <item> state is open, and the agent state is hand-free. As result, the <item> state will change to closed.
""".format(x)


def sg_2_plan(sg_exp: dict, sg_qry: dict, goal_exp: str, goal_qry: str,
              add_obj_exp: str = None, add_obj_qry: str = None, add_act_exp: str = None, add_act_qry: str = None):
    content = "You are an excellent graph planning agent. Given some domain knowledge and a scene graph representation of an environment, you can use it to generate a step-by-step task plan for solving a given goal instruction."

    info_add_obj_exp = ", {}". format(
        ", ".join(add_obj_exp)) if add_obj_exp is not None else ""
    info_add_act_exp = "and the following additional action(s): \n{}".format(
        "\n".join(add_act_exp)) if add_act_exp is not None else ""
    info_add_obj_qry = "there are the following new object type(s): {}.".format(
        ", ".join(add_obj_qry)) if add_obj_qry is not None else ""
    info_add_act_qry = "\nBesides the basic actions (goto, pick, drop), there are the following additional action(s): \n{}".format(
        "\n".join(add_act_qry)) if add_act_qry is not None else ""
    output_exp = sayplan_plan_exp()
    output_exp.pop("mode")

    prompt = f"""
    EXAMPLE:
    A domain has the following object types: agent, room, item{info_add_obj_exp}. 
    The agent can perform the following basic actions:
    goto(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>, where <room_1> and <room_2> should be neighbors. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be accessible and located in <room>, the pick action is in <item>'s affordance, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <item> is accessible, the drop action is in <item>'s affordance, <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    {info_add_act_exp}
    
    Here is an example of a scene graph in the form of a nested dictionary in Python:
    ```\n{sg_exp}\n```\n
    The top level keys are the name of the scene, the rooms, the agents, and possibly the humans.
    Each room contains a dictionary of 'items' inside the rooms, and a list of 'neighbor' (connected) rooms. The 'neighbor' relation is bidirectional, i.e. if kitchen is neighbor of corridor, then corridor is also neighbor of kitchen.
    Each item has three attributes, 'accessible' means if the item can be accessed or not, 'affordance' indicates the affordable actions of this item, 'state' infers whether the item is free, or occupied, e.g. being picked by an agent.
    Each agent has two attributes, the current position and the state.

    Given a goal description e.g., {goal_exp}, and using the previously defined object types and actions, you can generate the output according to the given OUTPUT RESPONSE FORMAT (Python dict, USE DOUBLE QUOTES!):
    {sayplan_output_format()}
    
    Following is an example output with respect to the example scene graph above:\n{output_exp}
    
    QUERY:
    Now in a new domain, {info_add_obj_qry} {info_add_act_qry}\n
    New instruction: {goal_qry}
    New 3D scene graph:\n{sg_qry}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent". Do not generate infinite actions in the plan!
    """
    return content, prompt


def sg_2_pddl(sg_example: dict, sg_query: dict, goal: str, add_action: str = None):
    # Load PDDL files
    pddl_path = os.path.join(os.getcwd(), "data/pddl/example/")
    domain_file = os.path.join(
        pddl_path, "{}_domain.pddl".format(sg_example["name"]))
    problem_file = os.path.join(
        pddl_path, "{}_problem.pddl".format(sg_example["name"]))
    if os.path.isfile(domain_file):
        with open(domain_file, "r") as df:
            domain = df.read()
    else:
        raise ("Cannot find domain file :{}".format(domain_file))
    if os.path.isfile(problem_file):
        with open(problem_file, "r") as pf:
            problem = pf.read()
    else:
        raise ("Cannot find problem file :{}".format(problem_file))

    content = "You are an excellent PDDL file generator. Given a scene graph representation of an environment, you can use it to generate a domain description file and a problem instance file in PDDL."
    prompt = p_template(dict(itertools.islice(sg_example.items(), 3))) + f"""
    Output Response Format:
    <Domain file> in PDDL, which describes the object types, the predicates, and the action knowledge (the preconditions and effects of an action).
    <Problem file> in PDDL, which describes the objects, the initial states and the goal specifications (consist of predicates defined in the domain file).
    
    Here is a pair of examples of a domain file and a problem file in PDDL with respect to the corresponding scene graph above:
    Domain file: \n{domain}\n
    Problem file: \n{problem}\n
    
    Now I have a new scene graph: \n{dict(itertools.islice(sg_query.items(), 3))}\n 
    {"Here the agent can perform the following additional action(s): {}.".format(add_action) if add_action is not None else None}
    
    I want to reach this goal: {goal}\n
    Please provide me a new domain file and a new problem file in PDDL with respect to the new scene and the goal specification directly without further explanations.
    """
    if add_action is not None:
        prompt += f"Here the agent can perform the following additional action(s): {add_action}"
    return content, prompt


def nl_2_pddl_domain(domain_exp: str, domain_qry_name: str, add_obj_exp: str = None, add_obj_qry: str = None,
                     add_act_exp: str = None, add_act_qry: str = None):
    content = "You are an excellent PDDL domain file generator. Given a description of object types, predicates, and action knowledge, you can use it to generate a PDDL domain description file."

    if add_obj_qry is None and add_act_qry is None:
        raise Exception(
            "Additional object and additional action cannot be both None!")
    info_add_obj_exp = ", {}". format(
        ", ".join(add_obj_exp)) if add_obj_exp is not None else ""
    info_add_act_exp = "and the following additional action(s): \n{}".format(
        "\n".join(add_act_exp)) if add_act_exp is not None else ""
    info_add_obj_qry = "there are the following new object type(s): {}.".format(
        ", ".join(add_obj_qry)) if add_obj_qry is not None else ""
    info_add_act_qry = "\nBesides the basic actions (goto, pick, drop), there are the following additional action(s): \n{}".format(
        "\n".join(add_act_qry)) if add_act_qry is not None else ""

    prompt = f"""
    For example, a domain has the following object types: agent, room, item{info_add_obj_exp}. 
    The agent can perform the following basic actions:
    goto(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>, where <room_1> and <room_2> should be neighbors. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be accessible and located in <room>, the pick action is in <item>'s affordance, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <item> is accessible, the drop action is in <item>'s affordance, <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    {info_add_act_exp}
    
    A PDDL domain file describes the object types, the predicates, and the action knowledge (the preconditions and effects of an action).
    The corresponding PDDL domain file with respect to the previous actions looks like: \n```\n{domain_exp}\n```
    with the first line of code defining the name of the domain.
    
    Now in a new domain named "{domain_qry_name}", {info_add_obj_qry} {info_add_act_qry}\n
    Please provide a new PDDL domain file with respect to this new domain and define the domain name as "{domain_qry_name}" directly without further explanations. Please also keep the comments such as "; Begin actions", "; End actions" etc. in the domain file.
    """

    return content, prompt


def nl_prune_item(items_exp: dict, items_qry: dict, goal_exp: str, goal_qry: str, item_keep_exp: list, domain_exp: str = None, domain_qry: str = None):
    act_exp, act_qry = None, None
    if domain_exp is not None:
        act_exp = "and the corresponding action knowledge\n{}".format(
            utils.get_pddl_domain_actions(domain_exp))
    if domain_qry is not None:
        act_qry = "and the new action knowledge\n{}".format(
            utils.get_pddl_domain_actions(domain_qry))

    content = "You are an excellent assistant in pruning items. Given a list of items and a goal description, you can prune the item list by only keeping the relevant items."
    prompt = f"""
    Here is an example of a list of items: {items_exp}
    
    Given an example of a goal description: {goal_exp}, {act_exp} 
    the relevant items for accomplishing the goal are {item_keep_exp}.
    
    Now given a new list of items: {items_qry}
    and a new goal description: {goal_qry}, {act_qry}
    please provide a list of the relevent items from the new item list for accomplishing the new goal directly without further explanations, and keep the same data structure.
    """

    return content, prompt


def sg_2_pddl_problem(domain_name_exp: str, domain_exp: str, problem_exp: str,
                      sg_exp: dict, sg_qry: dict, goal_exp: str, goal_qry: str,
                      domain_qry: str, domain_name_qry: str):
    content = "You are an excellent PDDL problem file generator. Given a scene graph representation of an environment, a PDDL domain file and a goal description, you can generate a PDDL problem file."
    prompt = f"""
    Here is an example of a scene graph in the form of a nested dictionary in Python:
    ```\n{sg_exp}\n```\n
    The top level keys are the name of the scene, the rooms, the agents, and possibly the humans.
    Each room contains a dictionary of 'items' inside the rooms, and a list of 'neighbor' (connected) rooms. The 'neighbor' relation is bidirectional, i.e. if kitchen is neighbor of corridor, then corridor is also neighbor of kitchen.
    Each item has three attributes, 'accessible' means if the item can be accessed or not, 'affordance' indicates the affordable actions of this item, 'state' infers whether the item is free, or occupied, e.g. being picked by an agent.
    Each agent has two attributes, the current position and the state.

    Given a goal description e.g., {goal_exp}, and using the pre-defined object types, predicated in the PDDL domain example named {domain_name_exp}:
    ```\n{utils.get_pddl_domain_types(domain_exp)}\n{utils.get_pddl_domain_predicates(domain_exp)}\n```
    A corresponding PDDL problem file can be formulated as follows:
    ```\n{problem_exp}\n```
    The first line defines the name of the problem, usually the scene graph's name.
    The second line refers to the domain it based on.
    The "(:objects )" section lists all the items included in the scene graph with corresponding object types. Remember to distinguish the additional object types from the other items.
    The "(:init )" section lists the connections (neighbors) of the rooms in the scene graph, the positions of the items and the agent, and the attributes of all listed items (e.g. accessible, pickable, turnable etc.).
    The ; Connections part lists the neighbor rooms of all rooms in the scene graph. Note that the connection of each two neighbor rooms always exists as a pair, e.g. if there is a "(neighbor corridor kitchen)", there should always exist a "(neighbor kitchen corridor)".
    The ; Positions part lists the positions of all items and the agent in the scene graph. 
    The ; Attributes part lists the attributes of all items in the scene graph.
    The "(:goal )" section defines the goal using the goal description given above.

    Now given a new scene graph: \n```\n{sg_qry}\n```
    and a new goal description: {goal_qry}
    and using the object types, predicates from the new PDDL domain file named {domain_name_qry}:
    ```\n{utils.get_pddl_domain_types(domain_qry)}\n{utils.get_pddl_domain_predicates(domain_qry)}\n```
    Please provide a new problem file in PDDL with respect to the new scene graph and goal specification directly without further explanations. Please also keep the comments such as "; Begin goal", "; End goal" etc. in the problem file.
    The goal should only consist of the previously defined predicates without any further keyword which not appear in the examples such as "forall" etc.
    """

    return content, prompt


def decompose_problem(goal_exp: str, subgoal_exp: list, subgoal_pddl_exp: list, item_keep_exp: list,
                      goal_qry: str, problem_exp: str, item_keep_qry: list, problem_qry: str,
                      domain_qry: str, acc_goal: bool = False):
    content = "You are an excellent assistant in decomposing long-term tasks. Given a task goal description and a corresponding PDDL problem file, you can decompose the task into multiple sub-tasks, and generate multiple PDDL sub-problem files correspondingly."
    subgoal_in_lines = ", \n".join(
        ["```" + sp + "```" for sp in subgoal_pddl_exp])
    if acc_goal:
        subgoal_pddl_exp_str = f"""When formulating the corresponding sub-goals in PDDL, to ensure that the previously achieved goals still remain valid, one can autoregressively append new sub-goal to the previous ones as follows:\n{subgoal_in_lines}"""
    else:
        subgoal_pddl_exp_str = f"""The corresponding sub-goal descriptions in PDDL can be formulated as: \n{subgoal_in_lines}"""

    prompt = f"""
    A PDDL problem file has three main sections:
    The "(:objects )" section lists all the object types in the environment.
    The "(:init )" section lists the 'Connections' (neighbors) of the rooms in the environment, the 'Positions' of the items and the agent, and the 'Attributes' of all listed items (e.g. accessible, pickable, turnable etc.).
    Note that the connection of each two neighbor rooms always exists as a pair, e.g. if there is a "neighbor corridor kitchen", there should always exist a "neighbor kitchen corridor".
    The "(:goal )" section defines the task goal.
    
    For example, given a goal description: {goal_exp}, and the following PDDL problem file:
    ```\n{problem_exp}\n```
    
    The task goal can first be broken down into multiple sub-goals: {subgoal_exp}.
    {subgoal_pddl_exp_str}
    
    Now given a new goal description: {goal_qry},
    and a corresponding new PDDL problem file:
    ```\n{problem_qry}n```
    and solely using the following predicates:
    ```\n{utils.get_pddl_domain_predicates(domain_qry)}\n```
    
    Note that the robot can only transport one item at a time.
    And the following item(s) are relevant for accomplishing the goal {item_keep_qry}.
    Please break down the new goal into a list of sub-goals as many as possible, and formulate them in PDDL (where each sub-goal only consists of one predicate). Use ``` to wrap each sub-goal in PDDL.
    """
    return content, prompt


def decompose_problem_chain(goal_exp: str, subgoal_exp: list, subgoal_pddl_exp: list, item_keep_exp: list,
                            goal_qry: str, problem_exp: str, item_keep_qry: list, problem_qry: str,
                            domain_qry: str, acc_goal: bool = False):
    content = "You are an excellent assistant in decomposing long-term tasks. Given a task goal description and a corresponding PDDL problem file, you can decompose the task into multiple sub-tasks, and generate multiple PDDL sub-problem files correspondingly."
    subgoal_in_lines = ", \n".join(
        ["```" + sp + "```" for sp in subgoal_pddl_exp])
    if acc_goal:
        subgoal_pddl_exp_str = f"""When formulating the corresponding sub-goals in PDDL, to ensure that the previously achieved goals still remain valid, one can autoregressively append new sub-goal to the previous ones as follows:\n{subgoal_in_lines}"""
    else:
        subgoal_pddl_exp_str = f"""The corresponding sub-goal descriptions in PDDL can be formulated as: \n{subgoal_in_lines}"""

    prompt = f"""
    For example, given a goal description: {goal_exp}, and the following PDDL problem file:
    ```\n{problem_exp}\n```
    
    The task goal can first be broken down into multiple sub-goals: {subgoal_exp}.
    {subgoal_pddl_exp_str}
    
    Now given a new goal description: {goal_qry},
    and a corresponding new PDDL problem file:
    ```\n{problem_qry}\n```
    and solely using the following predicates:
    ```\n{utils.get_pddl_domain_predicates(domain_qry)}\n```
    
    Note that the robot can only transport one item at a time.
    And the following item(s) are relevant for accomplishing the goal {item_keep_qry}.
    Please break down the new goal into a list of sub-goals as many as possible, and formulate them in PDDL (where each sub-goal only consists of one predicate). Use ``` to wrap each sub-goal in PDDL.
    """
    return content, prompt


def sayplan_search_prompt(goal_exp: str = None, goal_qry: str = None, sg_qry: dict = None):
    content = "You are an excellent graph exploration agent. Given a graph representation of an environment, you can explore the graph by expanding room nodes to find the items of interest."
    prompt = f"""
    ENVIRONMENT API:
    expand(<room>): Reveal items connected to a room node.
    contract(<room>): Hide items to reduce graph size for memory constraints.
    
    OUTPUT RESPONSE FORMAT:
    "mode": "search",
    "chain_of_thought": "break your problem down into a series of intermediate reasoning steps to help you determine your next command",
    "reasoning": "justify why the next action is important",
    "command": (command_name, room_name): "command_name": "expand" or "contract", "room_name": room to perform an operation on."
    
    EXAMPLE:\n{sayplan_search_exp(goal_exp)}
    
    QUERY:
    New instruction: {goal_qry}
    New 3D scene graph: \n{sg_qry}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent".
    """
    return content, prompt


def sayplan_plan_prompt(add_obj_exp: str = None, add_act_exp: str = None, add_state_exp: str = None,
                        add_obj_qry: str = None, add_act_qry: str = None, add_state_qry: str = None,
                        goal_exp: str = None, goal_qry: str = None, sg_exp: dict = None, sg_qry: dict = None):
    info_add_obj_exp = "\n* Additional object type(s): {}". format(
        ", ".join(add_obj_exp)) if add_obj_exp is not None else ""
    info_add_act_exp = "\n* Additional action(s): \n{}".format(
        "\n".join(add_act_exp)) if add_act_exp is not None else ""
    add_env_func_exp = f"Additional ENVIRONMENT FUNCTIONS:{info_add_obj_exp}{info_add_act_exp}"\
        if add_obj_exp is not None or add_act_exp is not None else ""
    add_env_state_exp = "Additional ENVIRONMENT STATES:\n{}".format(
        "\n".join(add_state_exp)) if add_state_exp is not None else ""

    info_add_obj_qry = "\n* New additional object type(s): {}".format(
        ", ".join(add_obj_qry)) if add_obj_qry is not None else ""
    info_add_act_qry = "\n* New additional action(s): \n{}".format(
        "\n".join(add_act_qry)) if add_act_qry is not None else ""
    add_env_func_qry = f"New additional ENVIRONMENT FUNCTIONS:{info_add_obj_qry}{info_add_act_qry}"\
        if add_obj_qry is not None or add_act_qry is not None else ""
    add_env_state_qry = "New additional ENVIRONMENT STATES:\n{}".format(
        "\n".join(add_state_qry)) if add_state_qry is not None else ""

    content = "You are an excellent graph planning agent. Given a graph representation of an environment, you can use this graph to generate a step-by-step task plan that the agent can follow to solve a given instruction."
    prompt = f"""
    We have now switched to planning mode.
    
    ENVIRONMENT FUNCTIONS:
    There are the following object types in the environment: agent, room, item.
    Note that robot is an instance of agent, and the robot can perform the following basic actions:
    goto(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>, where <room_1> and <room_2> should be neighbors. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be accessible and located in <room>, the pick action is in <item>'s affordance, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <item> is accessible, the drop action is in <item>'s affordance, <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    
    ENVIRONMENT STATES:
    neighbor(<room_1>, <room_2>): Room <room_1> and room <room_2> are neighboring to each other.
    agent_at(<agent>, <room>): <agent> is in <room>.
    item_at(<item>, <room>): <item> is in <room>.
    item_pickable(<item>): <item> can be picked up.
    item_accessible(<item>): <item> can be accessed.
    agent_loaded(<agent>): <agent> is loaded.
    agent_has_item(<agent>, <item>): <agent> has <item> in hand.

    OUTPUT RESPONSE FORMAT:
    "mode": "planning",
    "chain_of_thought": "break your problem down into a series of intermediate reasoning steps to help you determine your next command",
    "reasoning": "justify why the next action is important",
    "plan": "high-level task plan, which only consists of actions listed in the ENVIRONMENT FUNCTIONS above"

    EXAMPLE:
    Instruction: {goal_exp}
    3D scene graph: {sg_exp}
    {add_env_func_exp}
    {add_env_state_exp}
    Output: \n{sayplan_plan_exp()}
    
    QUERY:
    New instruction: {goal_qry}
    Previously explored 3D scene graph: \n{sg_qry}
    {add_env_func_qry}
    {add_env_state_qry}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent".
    """
    return content, prompt


def sayplan_prompt(add_obj_exp: str = None, add_act_exp: str = None, add_state_exp: str = None,
                   add_obj_qry: str = None, add_act_qry: str = None, add_state_qry: str = None,
                   goal_exp: str = None, sg_exp: str = None, output_exp: str = None,
                   goal_qry: str = None, sg_qry: str = None):
    info_add_obj_exp = "\n* Additional object type(s): {}". format(
        ", ".join(add_obj_exp)) if add_obj_exp is not None else ""
    info_add_act_exp = "\n* Additional action(s): \n{}".format(
        "\n".join(add_act_exp)) if add_act_exp is not None else ""
    add_env_func_exp = f"Additional ENVIRONMENT FUNCTIONS:{info_add_obj_exp}{info_add_act_exp}"\
        if add_obj_exp is not None or add_act_exp is not None else ""
    add_env_state_exp = "Additional ENVIRONMENT STATES:\n{}".format(
        "\n".join(add_state_exp)) if add_state_exp is not None else ""

    info_add_obj_qry = "\n* New additional object type(s): {}".format(
        ", ".join(add_obj_qry)) if add_obj_qry is not None else ""
    info_add_act_qry = "\n* New additional action(s): \n{}".format(
        "\n".join(add_act_qry)) if add_act_qry is not None else ""
    add_env_func_qry = f"New additional ENVIRONMENT FUNCTIONS:{info_add_obj_qry}{info_add_act_qry}"\
        if add_obj_qry is not None or add_act_qry is not None else ""
    add_env_state_qry = "New additional ENVIRONMENT STATES:\n{}".format(
        "\n".join(add_state_qry)) if add_state_qry is not None else ""

    content = "You are an excellent graph planning agent. Given a graph representation of an environment, you can use this graph to generate a step-by-step task plan that the agent can follow to solve a given instruction."
    prompt = f"""
    ENVIRONMENT FUNCTIONS:
    There are the following object types in the environment: agent, room, item.
    Note that robot is an instance of agent, and the robot can perform the following basic actions:
    goto(<agent>, <room_1>, <room_2>): <agent> goes from <room_1> to <room_2>, where <room_1> and <room_2> should be neighbors. As result, <agent> will leave <room_1> and locates in <room_2>.
    pick(<agent>, <item>, <room>): <agent> picks up an <item> at <room>. <item> must be accessible and located in <room>, the pick action is in <item>'s affordance, and <agent> state is hand-free. As result, <agent> state will change to loaded, and <item> left <room>.
    drop(<agent>, <item>, <room>): <agent> drops an <item> at <room>. <item> is accessible, the drop action is in <item>'s affordance, <agent> is in <room> and has <item> in hand, and <agent> state is loaded. As result, <item> will locate in <room>, <agent> state will change to hand-free.
    
    ENVIRONMENT STATES:
    neighbor(<room_1>, <room_2>): Room <room_1> and room <room_2> are neighboring to each other.
    agent_at(<agent>, <room>): <agent> is in <room>.
    item_at(<item>, <room>): <item> is in <room>.
    item_pickable(<item>): <item> can be picked up.
    item_accessible(<item>): <item> can be accessed.
    agent_loaded(<agent>): <agent> is loaded.
    agent_has_item(<agent>, <item>): <agent> has <item> in hand.
    
    OUTPUT RESPONSE FORMAT (Python dict, USE DOUBLE QUOTES!):\n{sayplan_output_format()}
    
    EXAMPLE:
    Instruction: {goal_exp}
    3D scene graph: {sg_exp}
    {add_env_func_exp}
    {add_env_state_exp}
    Output: \n{output_exp}
    
    QUERY:
    New instruction: {goal_qry}
    New 3D scene graph: \n{sg_qry}
    {add_env_func_qry}
    {add_env_state_qry}
    Please generate the output with according to the given OUTPUT RESPONSE FORMAT directly without further explanations. Make sure to use the term "robot" in the plan instead of "agent". Do not generate infinite actions in the plan!
    """
    return content, prompt


def sayplan_replan_prompt(err_info: str):
    content = "You are an excellent replanner. Given an error message, you can fix the previous plan to recover from the error."
    prompt = f"""
    Scene Graph Simulator (Feedback): {err_info}\n
    Fix the plan.
    """
    return content, prompt


def llmgenplan_domain_summary(domain_qry: str, problem_exps: list):
    problems = "\n".join(["Problem {}: \n{}".format(i, p)
                         for i, p in enumerate(problem_exps)])

    content = "You are an excellent PDDL domain summarizer. Given a PDDL domain file and a list of PDDL problem files, you can summarize the domain knowledge in words."
    prompt = f"""
    Domain:\n{domain_qry}
    
    Example problem(s):\n{problems}
    
    Write a short summary of this domain in words.
    """
    return content, prompt


def llmgenplan_strategy(domain_summary: str = None):
    domain = "Given a domain summary:\n{}\n".format(
        domain_summary) if domain_summary is not None else ""
    content = "You are an excellent strategy generator. Given a summary of the domain knowledge, you can generate a strategy to solve the problem."
    prompt = f"""{domain}There is a simple strategy for solving all problems in this domain without using search. What is that strategy?"""
    return content, prompt


def llmgenplan_impl_func(strategy: str = None, typed: bool = False):
    stg = "Given the following strategy: \n{}\n".format(
        strategy) if strategy is not None else ""
    object_description = "a set of (object name, type name) tuples" if typed else "a set of objects (string names)"
    content = "You are an excellent Python function writer. Given a strategy to solve the problem, you can implement Python functions to execute the strategy."
    prompt = f"""
    {stg}Implement the strategy as a Python function.

    The code should should be of the form

    ```python
    def get_plan(objects, init, goal):
        # Your code here
        return plan
    ```

    where
        - `objects` is {object_description}
        - `init` is a set of ground atoms represented as tuples of predicate
        names and arguments (e.g., ('predicate-foo', 'object-bar', ...))
        - `goal` is also a set of ground atoms represented in the same way
        - `plan` is a list of actions, where each action is a ground operator
        represented as a string (e.g., '(operator-baz object-qux ...)')
    
    Use ``` to wrap the code.
    """
    return content, prompt


def llmgenplan_replan(err_info: str = None):
    err = "Error: {}\n".format(err_info) if err_info is not None else ""
    content = "You are an excellent replanner. Given an error message, you can fix the Python code to recover from the error."
    prompt = f"{err}\nFix the code."
    return content, prompt
