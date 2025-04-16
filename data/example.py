HUMAN = {
    "scene": ["parole", "kemblesville"],
    "add_obj": ["human"],
    "add_act": [
        "pass(<agent>, <item>, <human>, <room>): <agent> passes an <item> to <human> at <room>. <item> is accessible, <agent> is in <room> and has <item> in hand, <agent> state is loaded, <human> is also in <room> but has nothing. As result, <human> will have <item>, and <agent> will have nothing in hand.",
        "make_coffee(<agent>, <item_1>, <item_2>, <room>): For making coffee, <item_1> must be mug and pickable, <item_2> must be coffee machine and turnable, both items are accessible, <agent> and both items should be in <room>, mug is not filled, and <agent> isn't loaded. As result, mug will be filled with coffee."
    ],
    "goal": "Make a cup of coffee with the mug and bring it to Tom.",
    "gt_cost": {
        "parole": 11,
        "kemblesville": 0  # FIXME
    },
    "item_keep": ["mug", "coffee_machine"],
    "subgoal": [
        "make coffee with mug",
        "bring coffee to Tom"
    ],
    "subgoal_pddl": [
        """
    (:goal\n        (mug_filled mug)\n    )\n""",
        """
    (:goal\n        (human_has_item tom mug)\n    )\n"""
    ]
}

LAUNDRY = {
    "scene": ["kemblesville", "parole"],
    "add_obj": None,
    "add_act": [
        "launder(<agent>, <item_1>, <item_2>, <item_3>, <room>): For laundering, <item_1> must be clothes and pickable, <item_2> must be detergent and pickable, <item_3> must be wash machine and turnable, all items are accessible, <agent> and all items should be in <room>, clothes are not clean, and <agent> isn't loaded. As result, clothes will be clean."
    ],
    "goal": "Launder the clothes and bring them to bedroom_1.",
    "gt_cost": {
        "kemblesville": 27,
        "parole": 0  # FIXME
    },
    "item_keep": ["clothes", "detergent", "wash_machine"],
    "subgoal": [
        "launder the clothes",
        "bring clothes to bedroom_1"
    ],
    "subgoal_pddl": [
        """
    (:goal
        (cloth_clean clothes)
    )\n""",
        """
    (:goal
        (and
            (cloth_clean clothes)
            (item_at clothes bedroom_1)
        )
    )\n"""
    ],
    "env_state": [
        "item_is_clothes(<item>): <item> is clothes.",
        "item_is_detergent(<item>): <item> is detergent.",
        "item_is_wash_machine(<item>): <item> is wash_machine.",
        "clothes_clean(<item>): <item> is clothes and is clean.",
    ]
}

PC = {
    "scene": ["allensville", "shelbiana", "parole"],
    "add_obj": ["pc"],
    "add_act": [
        "assemble(<agent>, <room>, <item_1>, <item_2>, <item_3>, <item_4>, <item_5>, <item_6>, <pc>): For assembling a pc, <agent> and all <item>s should be in <room>, <agent> isn't loaded and has none of the <item>s in hand, and <pc> is not assembled and therefore has no location, <item_1> must be mainboard, <item_2> must be cpu, <item_3> must be ram, <item_4> must be ssd, <item_5> must be gpu, <item_6> must be psu, all items are accessible and pickable. As result, <pc> is assembled."
    ],
    "goal": "Bring the necessary pc items to the living room and assemble my_pc.",
    "gt_cost": {
        "allensville": 41,
        "shelbiana": 42,
        "parole": 47
    },
    "item_keep": ["mainboard", "cpu", "ram", "ssd", "gpu", "psu"],
    "subgoal": [
        "bring mainboard to the living room",
        "bring cpu to the living room",
        "bring ram to the living room",
        "bring ssd to the living room",
        "bring gpu to the living room",
        "bring psu to the living room",
        "assemble my_pc"
    ],
    "subgoal_pddl": [
        """
    (:goal
        (item_at mainboard living_room)
    )\n""",
        """
    (:goal
        (item_at cpu living_room)
    )\n""",
        """
    (:goal
        (item_at ram living_room)
    )\n""",
        """
    (:goal
        (item_at ssd living_room)
    )\n""",
        """
    (:goal
        (item_at gpu living_room)
    )\n""",
        """
    (:goal
        (item_at psu living_room)
    )\n""",
        """
    (:goal
        (pc_assembled pc_1)
    )\n"""
    ],
    "env_state": [
        "item_is_mainboard(<item>): <item> is mainboard.",
        "item_is_cpu(<item>): <item> is cpu.",
        "item_is_ram(<item>): <item> is ram.",
        "item_is_ssd(<item>): <item> is ssd.",
        "item_is_gpu(<item>): <item> is gpu.",
        "item_is_psu(<item>): <item> is psu.",
        "pc_assembled(<item>): <item> is pc and is assembled.",
    ]
}


CLEAN = {
    "scene": ["shelbiana", "allensville", "parole"],
    "add_obj": None,
    "add_act": [
        "dispose(<agent>, <item_1>, <item_2>, <room>): For disposing, <item_1> must be pickable and accessible, <item_2> must be rubbish bin, <agent> and <item_2> should be in <room>, <agent> is loaded and has <item_1>, and <item_1> is not disposed. As result, <item_1> will be disposed, <agent> is not loaded and does not has <item_1> anymore, and battery will not be full.",
        "mop_floor(<agent>, <item>, <room>): For mopping floor, <item> must be mop and pickable, <agent> should be in <room>, <agent> is loaded and has <item>, mop is clean. As result, floor is clean in <room>, but mop will not be clean, and battery will not be full.",
        "clean_mop(<agent>, <item_1>, <item_2>, <room>): For cleaning mop, <item_1> must be mop and pickable, <item_2> must be sink, <agent> and <item_2> should be in <room>, <agent> is loaded and has <item_1>, and mop is not clean. As result, mop will be clean and lies in <room>, agent is not loaded and does not has mop anymore, and battery will not be full.",
        "charge(<agent>, <item>, <room>): For charging, <item> must be robot_hub and accessible, <agent> and <item> should be in <room>, <agent> is not loaded, and agent's battery is not full. As result, agent's battery will be full."
    ],
    "goal": "Identify and dispose the possible rubbish (e.g. food residue, drink bottles/cans etc.), mop the floor in living room and kitchen, note that all mops should be clean after mopping each room. The mop should be clean in the end, and the battery should be full.",
    "gt_cost": {
        "shelbiana": 43,
        "allensville": 39,
        "parole": 41
    },
    "item_keep": ["sink_1", "sink_2", "mop", "cola_can", "banana_peel", "rotting_apple", "rubbish_bin", "robot_hub"],
    "subgoal": [
        "Identify and dispose of the cola can",
        "Identify and dispose of the banana peel",
        "Identify and dispose of the rotting apple",
        "Mop the floor in the living room",
        "Clean the mop used for the living room",
        "Mop the floor in the kitchen",
        "Clean the mop used for the kitchen",
        "Charge the robot's battery to full"
    ],
    "subgoal_pddl": [
        """
    (:goal\n        (item_disposed cola_can)\n    )\n""",
        """
    (:goal\n        (item_disposed banana_peel)\n    )\n""",
        """
    (:goal\n        (item_disposed rotting_apple)\n    )\n""",
        """
    (:goal\n        (floor_clean living_room)\n    )\n""",
        """
    (:goal\n        (mop_clean mop)\n    )\n""",
        """
    (:goal\n        (floor_clean kitchen)\n    )\n""",
        """
    (:goal\n        (mop_clean mop)\n    )\n""",
        """
    (:goal\n        (battery_full robot)\n    )\n"""
    ],
    "env_state": [
        "item_is_mop(<item>): <item> is mop.",
        "item_is_sink(<item>): <item> is sink.",
        "item_is_rubbish_bin(<item>): <item> is rubbish_bin.",
        "item_is_robot_hub(<item>): <item> is robot_hub.",
        "item_disposed(<item>): <item> is disposed.",
        "floor_clean(<room>): floor in <room> is clean.",
        "mop_clean(<item>): <item> is mop and is clean.",
        "battery_full(<agent>): <agent>'s battery is full."
    ]
}


DINING = {
    "scene": ["parole", "shelbiana", "allensville"],
    "add_obj": None,
    "add_act": [
        "place_on(<agent>, <item_1>, <item_2>, <room>): For placing <item_1> on <item_2>, <agent> and <item_2> must be in <room>, <item_1> must be pickable and accessible, <item_2> must be dining_table, <agent> is loaded and has <item_1>. As result, <item_1> is on <item_2> in <room>, <agent> is not loaded and does not has <item_1> anymore."
    ],
    "goal": "Set up the dining table for dinner, place the tableware/cutleries and glass on the dining table. Also bring something romantic to the dining table.",
    "gt_cost": {
        "parole": 33,
        "shelbiana": 39,
        "allensville": 39
    },
    "item_keep": ["plate", "fork", "knife", "spoon", "glass", "flower", "dining_table"],
    "subgoal": [
        "Place the plate on the dining table",
        "Place the fork on the dining table",
        "Place the knife on the dining table",
        "Place the spoon on the dining table",
        "Place the glass on the dining table",
        "Place the flower on the dining table"
    ],
    "subgoal_pddl": [
        """
    (:goal\n        (item_on plate dining_table)\n    )\n""",
        """
    (:goal\n        (item_on fork dining_table)\n    )\n""",
        """
    (:goal\n        (item_on knife dining_table)\n    )\n""",
        """
    (:goal\n        (item_on spoon dining_table)\n    )\n""",
        """
    (:goal\n        (item_on glass dining_table)\n    )\n""",
        """
    (:goal\n        (item_on flower dining_table)\n    )\n"""
    ],
    "env_state": [
        "item_on(<item_1>, <item_2>): <item_1> is on <item_2>.",
        "item_is_dining_table(<item>): <item> is dining_table.",
    ]
}


OFFICE = {
    "scene": ["parole", "shelbiana", "allensville"],
    "add_obj": None,
    "add_act": [
        "pick_loadable() and drop_loadable(): similar to the pick() and drop() actions, but instead of judging whether the <item> is pickable, here the <item> should be loadable and empty.",
        "load(<agent>, <item_1>, <item_2>, <room>): For loading <item_2> into <item_1>, <agent> and <item_1> must be in <room>, <item_1> must be loadable and empty, <item_2> must be pickable is not in <item_1>, <agent> is loaded and has <item_2>. As result, <item_2> is in <item_1>, <item_2> is not in <room> anymore, <agent> is not loaded and does not has <item_2> anymore, and <item_1> is not empty anymore.",
        "unload(<agent>, <item_1>, <item_2>, <room>): For unloading <item_2> from <item_1>, <agent> and <item_1> must be in <room>, <item_1> must be loadable and not empty, <item_2> must be pickable and in <item_1>, <agent> is not loaded and does not have <item_2>. As result, <item_2> is not in <item_1>, <item_2> is in <room>, and <item_1> is empty."
    ],
    "goal": "Set up a home office in the living room by bringing the desk, lamp, locker and shelf along with the contents inside. Ensuring that the locker and shelf stay in living room when loading the contents. Note that you cannot move a loadable item if it is not empty!",
    "gt_cost": {
        "parole": 52,
        "shelbiana": 33,
        "allensville": 40
    },
    "item_keep": ["desk", "locker", "paper", "shelf", "book", "lamp"],
    "subgoal": [
        "Bring the desk to the living room",
        "Bring the lamp to the living room",
        "Bring the locker to the living room",
        "Bring the shelf to the living room",
        "Load the book into the shelf",
        "Load the paper into the locker"
    ],
    "subgoal_pddl": [
        """
    (:goal\n        (item_at desk living_room)\n    )\n""",
        """
    (:goal\n        (item_at lamp living_room)\n    )\n""",
        """
    (:goal\n        (item_at locker living_room)\n    )\n""",
        """
    (:goal\n        (item_at shelf living_room)\n    )\n""",
        """
    (:goal\n        (item_in book shelf)\n    )\n""",
        """
    (:goal\n        (item_in paper locker)\n    )\n"""
    ],
    "env_state": [
        "item_in(<item_1>, <item_2>): <item_2> is in <item_1>.",
        "item_loadable(<item>): <item> is loadable.",
    ]
}


def get_example(domain: str, scene: str = None):
    return eval(domain.upper())


def get_scenes(domain: str):
    return eval(domain.upper())["scene"]
