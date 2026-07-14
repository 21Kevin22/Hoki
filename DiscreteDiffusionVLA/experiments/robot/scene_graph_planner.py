"""Minimal scene-graph planner for converting graph goals into action plans."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_ACTIONS = [
    'pick',
    'place_in',
    'place_on',
    'place_under',
    'place_next_to',
    'place_left_of',
    'place_right_of',
    'place_in_front_of',
    'place_behind',
    'move_to',
    'lower_to_grasp_height',
    'micro_adjust_xy',
    'open',
    'close',
    'turn_on',
    'turn_off',
    'push_to',
    'pull',
    'press',
    'achieve_relation',
    'achieve_state',
]


PLANNER_INPUT_SCHEMA = {
    "instruction": "Natural-language task instruction.",
    "current_scene_graph": {
        "objects": [
            {
                "id": "Unique object id, e.g. red_block_1.",
                "label": "Optional human-readable name.",
                "class": "Optional semantic class.",
            }
        ],
        "relations": [
            {
                "type": "Relation name such as on, in, holding.",
                "subject": "Source object id.",
                "object": "Target object id.",
            }
        ],
        "states": [
            {
                "object": "Object id.",
                "state": "State name such as open, closed, turned_on.",
            }
        ],
    },
    "goal_scene_graph": "Scene graph with the same schema describing the desired end state.",
}



SPECIAL_PREGRASP_OBJECTS = {
    'cream_cheese_1',
    'butter_1',
    'chocolate_pudding_1',
}

SIDE_GRASP_OBJECTS = {
    'cream_cheese_1': [
        'pinch the sides of the {object_name}, close the gripper gently, and lift it straight up',
        'grasp the {object_name} securely from the sides and raise it above the table',
        'move down to the {object_name}, clamp it carefully, and lift it up',
        'close the gripper around the {object_name} box and pick it up without pushing it away',
    ],
    'butter_1': [
        'grasp the butter pack gently from the sides and lift it straight up',
        'close the gripper carefully around the butter and raise it without sliding it away',
        'move down onto the butter, pinch it lightly, and lift it up',
        'secure the butter from the sides and hold it above the table',
    ],
    'chocolate_pudding_1': [
        'pinch the chocolate pudding cup gently and lift it straight up',
        'grasp the chocolate pudding carefully from the sides and raise it above the table',
        'move down to the chocolate pudding cup, close the gripper lightly, and lift it',
        'pick up the chocolate pudding cup securely without pushing it sideways',
    ],
}

TOP_GRASP_OBJECTS = {
    'milk_1': [
        'align over the milk carton, close the gripper firmly around it, and lift it up',
        'grasp the milk carton securely from above and raise it off the table',
        'move down onto the milk carton, close the gripper, and lift straight up',
        'pick up the milk carton and hold it steady above the table',
    ],
    'salad_dressing_1': [
        'align over the salad dressing bottle, close the gripper around its body, and lift it',
        'grasp the salad dressing bottle firmly and raise it straight up',
        'move down to the salad dressing bottle, secure it, and lift it off the table',
        'pick up the salad dressing bottle and hold it upright above the table',
    ],
    'tomato_sauce_1': [
        'align over the tomato sauce bottle, close the gripper around it, and lift it up',
        'grasp the tomato sauce bottle firmly and raise it off the table',
        'move down to the tomato sauce bottle, secure it, and lift straight up',
        'pick up the tomato sauce bottle and keep it upright above the table',
    ],
    'ketchup_1': [
        'align over the ketchup bottle, close the gripper around it, and lift it up',
        'grasp the ketchup bottle firmly and raise it off the table',
        'move down to the ketchup bottle, secure it, and lift straight up',
        'pick up the ketchup bottle and keep it upright above the table',
    ],
    'bbq_sauce_1': [
        'align over the bbq sauce bottle, close the gripper around it, and lift it up',
        'grasp the bbq sauce bottle firmly and raise it off the table',
        'move down to the bbq sauce bottle, secure it, and lift straight up',
        'pick up the bbq sauce bottle and keep it upright above the table',
    ],
    'orange_juice_1': [
        'align over the orange juice carton, close the gripper around it, and lift it up',
        'grasp the orange juice carton firmly and raise it off the table',
        'move down to the orange juice carton, secure it, and lift straight up',
        'pick up the orange juice carton and keep it upright above the table',
    ],
}

APPROACH_OBJECT_PROMPTS = {
    'cream_cheese_1': [
        'position the gripper tightly over the center of the {target_name} box for grasping',
        'align the gripper with the narrow sides of the {target_name} from directly above',
        'move closer above the {target_name} and center the gripper before closing',
        'hover just above the {target_name} so the gripper can pinch it cleanly',
    ],
    'butter_1': [
        'center the gripper carefully over the butter pack from directly above',
        'align the gripper with the middle of the butter so it can be pinched cleanly',
        'hover just above the butter and square the gripper before grasping',
        'move directly over the center of the butter pack for a gentle side grasp',
    ],
    'milk_1': [
        'position the gripper over the center of the milk carton for a stable grasp',
        'align the gripper with the broad faces of the milk carton from above',
        'hover above the milk carton and center the gripper before closing',
        'move directly over the milk carton for a firm top-down grasp',
    ],
    'salad_dressing_1': [
        'position the gripper over the center of the salad dressing bottle',
        'align the gripper with the body of the salad dressing bottle from above',
        'hover above the salad dressing bottle and center the gripper before closing',
        'move directly over the salad dressing bottle for a stable grasp',
    ],
    'tomato_sauce_1': [
        'position the gripper over the center of the tomato sauce bottle',
        'align the gripper with the body of the tomato sauce bottle from above',
        'hover above the tomato sauce bottle and center the gripper before closing',
        'move directly over the tomato sauce bottle for a stable grasp',
    ],
}

SIDE_APPROACH_OBJECT_PROMPTS = {
    'butter_1': [
        'approach the butter pack from the side and align the gripper with its long edge before grasping',
        'move to the side of the butter and square the gripper with the narrow faces for a side pinch',
        'position the gripper beside the butter pack so it can close around the sides without pushing it forward',
        'bring the gripper in from the side of the butter and line it up for a gentle side grasp',
    ],
}

PLANNER_OUTPUT_SCHEMA = {
    "schema_version": "scene_graph_plan.v1",
    "instruction": "Original user instruction.",
    "subgoals": [
        {
            "subgoal_id": "Unique subgoal id.",
            "goal_reference": "Relation/state being solved in this stage.",
            "kind": "relation or state",
            "status": "pending or already_satisfied",
        }
    ],
    "plan": [
        {
            "step_id": "Unique step id.",
            "action": "Planner action name such as pick, place_in, open.",
            "args": {"object": "Bound object ids", "target": "Optional target id."},
            "instruction_text": "Natural-language rendering of the action.",
            "goal_reference": "Goal relation/state that motivated this step.",
            "status": "pending",
        }
    ],
    "next_action": "First step from plan, or null if already satisfied.",
    "goal_satisfied": False,
}


def plan_from_scene_graph_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    instruction = _clean_text(payload.get('instruction'))
    current_scene_graph = _normalize_scene_graph(payload.get('current_scene_graph'))
    goal_scene_graph = _normalize_scene_graph(payload.get('goal_scene_graph'))

    current_objects = _object_lookup(current_scene_graph)
    goal_objects = _object_lookup(goal_scene_graph)
    object_lookup = {**current_objects, **goal_objects}

    current_relations = set(_normalize_relation_list(current_scene_graph.get('relations', [])))
    goal_relations = list(_normalize_relation_list(goal_scene_graph.get('relations', [])))

    current_states = set(_normalize_state_list(current_scene_graph.get('states', [])))
    goal_states = list(_normalize_state_list(goal_scene_graph.get('states', [])))

    subgoals = _build_decomposed_subgoals(goal_relations, goal_states, current_relations, current_states)

    steps: List[Dict[str, Any]] = []
    resolved_subgoals: List[Dict[str, Any]] = []
    step_idx = 1
    symbolic_relations = set(current_relations)
    symbolic_states = set(current_states)

    for subgoal_idx, subgoal in enumerate(subgoals, start=1):
        goal_reference = subgoal['goal_reference']
        if subgoal['kind'] == 'relation':
            planned_steps = _plan_for_relation(goal_reference, symbolic_relations, symbolic_states, object_lookup)
        else:
            planned_steps = _plan_for_state(goal_reference, object_lookup)

        subgoal_id = f"subgoal_{subgoal_idx:03d}"
        resolved_subgoals.append({
            'subgoal_id': subgoal_id,
            'goal_reference': _format_goal_reference(goal_reference),
            'kind': subgoal['kind'],
            'status': 'pending' if planned_steps else 'already_satisfied',
        })

        for step in planned_steps:
            step_record = _step_record(step_idx, step, goal_reference, subgoal_id=subgoal_id)
            steps.append(step_record)
            _apply_symbolic_action_effect(step, symbolic_relations, symbolic_states)
            step_idx += 1

    response = {
        'schema_version': 'scene_graph_plan.v1',
        'planner_input_schema': PLANNER_INPUT_SCHEMA,
        'planner_output_schema': PLANNER_OUTPUT_SCHEMA,
        'instruction': instruction,
        'current_scene_graph': current_scene_graph,
        'goal_scene_graph': goal_scene_graph,
        'subgoals': resolved_subgoals,
        'plan': steps,
        'next_action': steps[0] if steps else None,
        'goal_satisfied': len(steps) == 0,
        'num_steps': len(steps),
    }
    return response


def _normalize_scene_graph(scene_graph: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(scene_graph, dict):
        return {'objects': [], 'relations': [], 'states': []}
    return {
        'objects': list(scene_graph.get('objects', []) or []),
        'relations': list(scene_graph.get('relations', []) or []),
        'states': list(scene_graph.get('states', []) or []),
    }


def _object_lookup(scene_graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for obj in scene_graph.get('objects', []):
        if isinstance(obj, dict) and _clean_text(obj.get('id')):
            lookup[obj['id']] = obj
    return lookup


def _normalize_relation_list(relations: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    normalized: List[Tuple[str, str, str]] = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rel_type = _clean_text(rel.get('type')).lower()
        subject = _clean_text(rel.get('subject'))
        obj = _clean_text(rel.get('object'))
        if rel_type and subject and obj:
            normalized.append((rel_type, subject, obj))
    return normalized


def _normalize_state_list(states: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    normalized: List[Tuple[str, str]] = []
    for state in states:
        if not isinstance(state, dict):
            continue
        obj = _clean_text(state.get('object'))
        name = _clean_text(state.get('state')).lower()
        if obj and name:
            normalized.append((obj, name))
    return normalized


def _plan_for_relation(
    relation: Tuple[str, str, str],
    current_relations: Sequence[Tuple[str, str, str]],
    current_states: Sequence[Tuple[str, str]],
    object_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rel_type, subject, target = relation
    del current_relations, current_states
    if rel_type in {'in', 'inside'}:
        return _transport_steps(subject, target, 'place_in', object_lookup)
    if rel_type == 'on':
        return _transport_steps(subject, target, 'place_on', object_lookup)
    if rel_type == 'under':
        return _transport_steps(subject, target, 'place_under', object_lookup)
    if rel_type == 'next_to':
        return _transport_steps(subject, target, 'place_next_to', object_lookup)
    if rel_type == 'left_of':
        return _transport_steps(subject, target, 'place_left_of', object_lookup)
    if rel_type == 'right_of':
        return _transport_steps(subject, target, 'place_right_of', object_lookup)
    if rel_type in {'in_front_of', 'front_of'}:
        return _transport_steps(subject, target, 'place_in_front_of', object_lookup)
    if rel_type == 'behind':
        return _transport_steps(subject, target, 'place_behind', object_lookup)
    if rel_type == 'holding':
        target_object = target if subject.startswith('robot') else subject
        return [
            _make_action('move_to', {'target': target_object}, object_lookup),
            _make_action('pick', {'object': target_object}, object_lookup),
        ]
    if rel_type == 'reachable':
        return [_make_action('move_to', {'target': target}, object_lookup)]
    if rel_type == 'pressed':
        return [_make_action('press', {'object': subject}, object_lookup)]
    if rel_type == 'pulled_to':
        return [_make_action('move_to', {'target': subject}, object_lookup), _make_action('pull', {'object': subject, 'target': target}, object_lookup)]
    if rel_type == 'pushed_to':
        return [_make_action('move_to', {'target': subject}, object_lookup), _make_action('push_to', {'object': subject, 'target': target}, object_lookup)]
    return [_make_action('achieve_relation', {'relation': rel_type, 'subject': subject, 'target': target}, object_lookup)]


def _transport_steps(
    subject: str,
    target: str,
    place_action: str,
    object_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if subject == 'butter_1':
        steps = [
            _make_action('move_to', {'target': subject, 'phase': 'approach_object_side'}, object_lookup),
            _make_action('micro_adjust_xy', {'target': subject, 'phase': 'pre_lower_side_alignment'}, object_lookup),
            _make_action('lower_to_grasp_height', {'target': subject}, object_lookup),
            _make_action('lower_to_grasp_height', {'target': subject, 'phase': 'refine_grasp_height'}, object_lookup),
        ]
    else:
        steps = [
            _make_action('move_to', {'target': subject, 'phase': 'approach_object'}, object_lookup),
        ]
        if subject in SPECIAL_PREGRASP_OBJECTS:
            steps.append(_make_action('lower_to_grasp_height', {'target': subject}, object_lookup))
            steps.append(_make_action('micro_adjust_xy', {'target': subject}, object_lookup))
    steps.append(_make_action('pick', {'object': subject}, object_lookup))
    if place_action.startswith('place_'):
        steps.append(_make_action('move_to', {'target': target, 'phase': 'approach_target'}, object_lookup))
    steps.append(_make_action(place_action, {'object': subject, 'target': target}, object_lookup))
    return steps


def _build_decomposed_subgoals(
    goal_relations: Sequence[Tuple[str, str, str]],
    goal_states: Sequence[Tuple[str, str]],
    current_relations: Sequence[Tuple[str, str, str]],
    current_states: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    current_relations = set(current_relations)
    current_states = set(current_states)
    relation_subgoals = [
        {'kind': 'relation', 'goal_reference': rel}
        for rel in goal_relations
        if rel not in current_relations
    ]
    state_subgoals = [
        {'kind': 'state', 'goal_reference': state}
        for state in goal_states
        if state not in current_states
    ]

    def relation_priority(item: Dict[str, Any]) -> Tuple[int, str, str]:
        rel_type, subject, target = item['goal_reference']
        priority = {
            'holding': 0,
            'in': 2,
            'inside': 2,
            'on': 3,
            'under': 4,
            'next_to': 5,
            'left_of': 6,
            'right_of': 6,
            'in_front_of': 7,
            'front_of': 7,
            'behind': 7,
        }.get(rel_type, 10)
        return (priority, target, subject)

    def state_priority(item: Dict[str, Any]) -> Tuple[int, str]:
        obj, state_name = item['goal_reference']
        priority = {'open': 0, 'opened': 0, 'turn_on': 1, 'turned_on': 1}.get(state_name, 5)
        return (priority, obj)

    state_subgoals.sort(key=state_priority)
    relation_subgoals.sort(key=relation_priority)
    return state_subgoals + relation_subgoals


def _apply_symbolic_action_effect(
    action: Dict[str, Any],
    current_relations: set[Tuple[str, str, str]],
    current_states: set[Tuple[str, str]],
) -> None:
    action_name = action.get('action')
    args = action.get('args', {})
    placement_relations = {'in', 'inside', 'on', 'under', 'next_to', 'left_of', 'right_of', 'in_front_of', 'behind'}

    if action_name == 'pick':
        obj = args.get('object')
        if not obj:
            return
        current_relations.difference_update({rel for rel in list(current_relations) if rel[1] == obj and rel[0] in placement_relations})
        current_relations.add(('holding', 'robot_1', obj))
        return

    if action_name and action_name.startswith('place_'):
        obj = args.get('object')
        target = args.get('target')
        if not obj:
            return
        current_relations.discard(('holding', 'robot_1', obj))
        current_relations.difference_update({rel for rel in list(current_relations) if rel[1] == obj and rel[0] in placement_relations})
        rel_map = {
            'place_in': 'in',
            'place_on': 'on',
            'place_under': 'under',
            'place_next_to': 'next_to',
            'place_left_of': 'left_of',
            'place_right_of': 'right_of',
            'place_in_front_of': 'in_front_of',
            'place_behind': 'behind',
        }
        rel_type = rel_map.get(action_name)
        if rel_type and target:
            current_relations.add((rel_type, obj, target))
        return

    if action_name in {'lower_to_grasp_height', 'micro_adjust_xy'}:
        return

    if action_name == 'open' and args.get('object'):
        obj = args['object']
        current_states.discard((obj, 'closed'))
        current_states.add((obj, 'open'))
        return

    if action_name == 'close' and args.get('object'):
        obj = args['object']
        current_states.discard((obj, 'open'))
        current_states.add((obj, 'closed'))
        return

    if action_name == 'turn_on' and args.get('object'):
        obj = args['object']
        current_states.discard((obj, 'turned_off'))
        current_states.add((obj, 'turned_on'))
        return

    if action_name == 'turn_off' and args.get('object'):
        obj = args['object']
        current_states.discard((obj, 'turned_on'))
        current_states.add((obj, 'turned_off'))


def _plan_for_state(state: Tuple[str, str], object_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    obj, state_name = state
    mapping = {
        'open': 'open',
        'opened': 'open',
        'closed': 'close',
        'close': 'close',
        'turned_on': 'turn_on',
        'on': 'turn_on',
        'turned_off': 'turn_off',
        'off': 'turn_off',
        'pressed': 'press',
        'pulled': 'pull',
    }
    action = mapping.get(state_name)
    if action is None:
        return [_make_action('achieve_state', {'object': obj, 'state': state_name}, object_lookup)]
    return [_make_action(action, {'object': obj}, object_lookup)]


def _step_record(
    step_idx: int,
    action: Dict[str, Any],
    goal_reference: Tuple[Any, ...],
    subgoal_id: str | None = None,
) -> Dict[str, Any]:
    step = dict(action)
    step['step_id'] = f'step_{step_idx:03d}'
    step['goal_reference'] = _format_goal_reference(goal_reference)
    step['subgoal_id'] = subgoal_id
    step['status'] = 'pending'
    return step


def _format_goal_reference(goal_reference: Tuple[Any, ...]) -> str:
    return ':'.join(str(x) for x in goal_reference)


def _make_action(action: str, args: Dict[str, Any], object_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    instruction_candidates = _instruction_candidates(action, args, object_lookup)
    return {
        'action': action,
        'args': args,
        'instruction_text': instruction_candidates[0],
        'instruction_candidates': instruction_candidates,
    }


def _action_to_instruction(action: str, args: Dict[str, Any], object_lookup: Dict[str, Dict[str, Any]]) -> str:
    return _instruction_candidates(action, args, object_lookup)[0]


def _instruction_candidates(action: str, args: Dict[str, Any], object_lookup: Dict[str, Dict[str, Any]]) -> List[str]:
    def label(object_id: str) -> str:
        obj = object_lookup.get(object_id, {})
        return _clean_text(obj.get('label')) or object_id.replace('_', ' ')

    if action == 'pick':
        object_name = label(args['object'])
        object_id = args['object']
        if object_id in SIDE_GRASP_OBJECTS:
            return [template.format(object_name=object_name) for template in SIDE_GRASP_OBJECTS[object_id]]
        if object_id in TOP_GRASP_OBJECTS:
            return [template.format(object_name=object_name) for template in TOP_GRASP_OBJECTS[object_id]]
        return [
            f"close the gripper around the {object_name} and lift it up",
            f"move down to grasp the {object_name}, close the gripper, and raise it",
            f"grasp the {object_name} firmly and lift it up",
            f"securely pick up the {object_name} and hold it above the table",
        ]
    if action == 'place_in':
        object_name = label(args['object'])
        target_name = label(args['target'])
        return [
            f"release the {object_name} into the {target_name}",
            f"lower the {object_name} into the {target_name} and open the gripper",
            f"place the {object_name} in the {target_name} and let go",
            f"set the {object_name} down inside the {target_name} and release it",
        ]
    if action == 'place_on':
        return [f"place the {label(args['object'])} on the {label(args['target'])}"]
    if action == 'place_under':
        return [f"place the {label(args['object'])} under the {label(args['target'])}"]
    if action == 'place_next_to':
        return [f"place the {label(args['object'])} next to the {label(args['target'])}"]
    if action == 'place_left_of':
        return [f"place the {label(args['object'])} to the left of the {label(args['target'])}"]
    if action == 'place_right_of':
        return [f"place the {label(args['object'])} to the right of the {label(args['target'])}"]
    if action == 'place_in_front_of':
        return [f"place the {label(args['object'])} in front of the {label(args['target'])}"]
    if action == 'place_behind':
        return [f"place the {label(args['object'])} behind the {label(args['target'])}"]
    if action == 'open':
        return [f"open the {label(args['object'])}"]
    if action == 'close':
        return [f"close the {label(args['object'])}"]
    if action == 'turn_on':
        return [f"turn on the {label(args['object'])}"]
    if action == 'turn_off':
        return [f"turn off the {label(args['object'])}"]
    if action == 'push_to':
        return [f"push the {label(args['object'])} toward the {label(args['target'])}"]
    if action == 'pull':
        target = args.get('target')
        if target:
            return [f"pull the {label(args['object'])} toward the {label(target)}"]
        return [f"pull the {label(args['object'])}"]
    if action == 'press':
        return [f"press the {label(args['object'])}"]
    if action == 'move_to':
        target_name = label(args['target'])
        target_id = args.get('target')
        phase = args.get('phase')
        if phase == 'approach_target':
            return [
                f"move the held object directly above the {target_name}",
                f"position the gripper over the center of the {target_name}",
                f"bring the object above the {target_name} for release",
            ]
        if phase == 'approach_object' and target_id in APPROACH_OBJECT_PROMPTS:
            return [template.format(target_name=target_name) for template in APPROACH_OBJECT_PROMPTS[target_id]]
        if phase == 'approach_object_side' and target_id in SIDE_APPROACH_OBJECT_PROMPTS:
            return [template.format(target_name=target_name) for template in SIDE_APPROACH_OBJECT_PROMPTS[target_id]]
        return [
            f"move the gripper directly above the {target_name}",
            f"approach the {target_name} carefully from above",
            f"align the gripper with the {target_name} for grasping",
        ]
    if action == 'lower_to_grasp_height':
        target_name = label(args['target'])
        return [
            f"lower the gripper straight down to the grasp height of the {target_name}",
            f"move slightly down onto the {target_name} so the gripper is ready to close",
            f"descend carefully to the grasp height of the {target_name} without pushing it away",
        ]
    if action == 'micro_adjust_xy':
        target_name = label(args['target'])
        phase = args.get('phase')
        if phase == 'pre_lower_side_alignment':
            return [
                f"make a small sideways adjustment so the gripper is centered on the sides of the {target_name}",
                f"shift horizontally to line the gripper up with the side faces of the {target_name}",
                f"nudge sideways until the gripper is aligned for a side grasp on the {target_name}",
            ]
        return [
            f"make a small horizontal adjustment to center the gripper over the {target_name}",
            f"shift slightly in the x y plane to align with the center of the {target_name}",
            f"nudge sideways to line the gripper up precisely over the {target_name}",
        ]
    if action == 'achieve_relation':
        return [(
            f"achieve relation {args['relation']} between {label(args['subject'])} "
            f"and {label(args['target'])}"
        )]
    if action == 'achieve_state':
        return [f"set the {label(args['object'])} to state {args['state']}"]
    return [action.replace('_', ' ')]


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ''
