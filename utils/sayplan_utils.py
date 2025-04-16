# Self-implemented utils code for the baseline approach SayPlan due to the lack of available open-source code
# Original paper: K. Rana et al., CoRL 2023, "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning"
# Paper link: https://proceedings.mlr.press/v229/rana23a/rana23a.pdf
# Implementation based on the prompt structure and input-output examples in the appendix of the paper


from data.scene_graph import collapsed_sg, update_sg, KEMBLESVILLE
import sys

sys.path.append("..")


def sayplan_output_format():
    return {
        "chain_of_thought": "break your problem down into a series of intermediate reasoning steps to help you determine your next command",
        "reasoning": "justify why the next action is important",
        "plan": "high-level task plan, which only consists of actions listed in the ENVIRONMENT FUNCTIONS above. The number of actions (plan length) is always finite."
    }


def sayplan_search_exp(goal: str):
    # Example prompt for the search task using laundry domain and kemblesville scene
    csg = collapsed_sg(KEMBLESVILLE)
    output = f"""
    Instruction: {goal}
    3D scene graph: {csg}
    Memory (history of previously expanded room nodes): []
    Output: {{
        "mode": "search",
        "chain_of_thought": "Explore graph for identifiying relevant items for solving the given task -> perform common sense exploration first -> explore kitchen -> explore bathroom -> explore bedroom_1 -> if items not found then explore other nodes.",
        "reasoning": "I will expand kitchen.",
        "command": ("expand", "kitchen")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "kitchen")}
    Memory: ["kitchen"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "I have found wash_machine in kitchen which is relevant for solving the task -> leave this room expanded -> still further items missing for the task -> bathroom might have relevant items -> explore bathroom next.",
        "reasoning": "I will expand bathroom next.",
        "command": ("expand", "bathroom")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "bathroom")}
    Memory: ["kitchen", "bathroom"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "I have found detergent in bathroom -> leave this room expanded -> still further items missing for the task -> bedroom_1 might have clothes -> explore bedroom_1 next.",
        "reasoning": "I will expand bedroom_1 next.",
        "command": ("expand", "bedroom_1")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "bedroom_1")}
    Memory: ["kitchen", "bathroom", "bedroom_1"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "bedroom_1 does not have clothes or anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract bedroom_1 to save memory.",
        "command": ("contract", "bedroom_1")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "contract", "bedroom_1")}
    Memory: ["kitchen", "bathroom", "bedroom_1"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "closet_1, closet_2, corridor_1, corridor_2, bedroom_2, and living_room are unexplored yet -> I will try bedroom_2 next.",
        "reasoning": "I will expand closet_1 as it is unexplored.",
        "command": ("expand", "bedroom_2")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "bedroom_2")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "bedroom_1 does not have clothes or anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract bedroom_2 to save memory.",
        "command": ("contract", "bedroom_2")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "contract", "bedroom_2")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "closet_1, closet_2, corridor_1, corridor_2, and living_room are unexplored yet -> I will try closet_1 next.",
        "reasoning": "I will expand closet_1 as it is unexplored.",
        "command": ("expand", "closet_1")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "closet_1")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "I have found clothes in closet_1 -> leave this room expanded -> closet_2, corridor_1, corridor_2, and living_room are unexplored yet -> I will try closet_2 next.",
        "reasoning": "I will expand closet_2 as it is unexplored.",
        "command": ("expand", "closet_2")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "closet_2")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "closet_2 does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract closet_2 to save memory.",
        "command": ("contract", "closet_2")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "contract", "closet_2")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "corridor_1, corridor_2, and living_room are unexplored yet -> I will try corridor_1 next.",
        "reasoning": "I will expand corridor_1 as it is unexplored.",
        "command": ("expand", "corridor_1")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "corridor_1")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "corridor_1 does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract corridor_1 to save memory.",
        "command": ("contract", "corridor_1")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "contract", "corridor_1")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "corridor_2 and living_room are unexplored yet -> I will try corridor_2 next.",
        "reasoning": "I will expand corridor_2 as it is unexplored.",
        "command": ("expand", "corridor_2")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "corridor_2")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "corridor_2 does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I will explore other unexplored rooms after doing this.",
        "reasoning": "I will contract corridor_2 to save memory.",
        "command": ("contract", "corridor_2")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "contract", "corridor_2")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "living_room is unexplored yet -> I will try living_room next.",
        "reasoning": "I will expand living_room as it is unexplored.",
        "command": ("expand", "living_room")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "expand", "living_room")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2", "living_room"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "living_room does not have anything relevant towards solving this task -> I should contract this room to save token memory -> I have explored all rooms and found all relevant items for solving the task.",
        "reasoning": "I will contract living_room to save memory.",
        "command": ("contract", "living_room")
    }}
    3D scene graph: {update_sg(csg, KEMBLESVILLE, "contract", "living_room")}
    Memory: ["kitchen", "bathroom", "bedroom_1", "bedroom_2", "closet_1", "closet_2", "corridor_1", "corridor_2", "living_room"]
    Output: {{
        "mode": "search",
        "chain_of_thought": "I have found all relevant items for solving the task -> search complete -> switch to planning mode.",
        "reasoning": "I will switch to planning mode.",
        "command": "Switch to planning"
    }}
    """
    return output


def sayplan_plan_exp():
    # Example prompt for the planning task using laundry domain and kemblesville scene
    output = {
        "mode": "planning",
        "chain_of_thought": "I have located the relevent items: clothes, detergent, and wash_machine -> generate plan for launder the clothes -> launder the clothes with detergent and wash_machine and bring them to bedroom_1.",
        "reasoning": "I will generate a task plan using the identified scene graph.",
        "plan": """(goto robot living_room corridor_1)
            (goto robot corridor_1 bathroom)
            (pick robot detergent bathroom)
            (goto robot bathroom corridor_1)
            (goto robot corridor_1 living_room)
            (goto robot living_room corridor_2)
            (goto robot corridor_2 kitchen)
            (drop robot detergent kitchen)
            (goto robot kitchen corridor_2)
            (goto robot corridor_2 living_room)
            (goto robot living_room corridor_1)
            (goto robot corridor_1 bedroom_1)
            (goto robot bedroom_1 closet_1)
            (pick robot clothes closet_1)
            (goto robot closet_1 bedroom_1)
            (goto robot bedroom_1 corridor_1)
            (goto robot corridor_1 living_room)
            (goto robot living_room corridor_2)
            (goto robot corridor_2 kitchen)
            (drop robot clothes kitchen)
            (launder robot clothes detergent wash_machine kitchen)
            (pick robot clothes kitchen)
            (goto robot kitchen corridor_2)
            (goto robot corridor_2 living_room)
            (goto robot living_room corridor_1)
            (goto robot corridor_1 bedroom_1)
            (drop robot clothes bedroom_1)  
        """
    }
    return output
