"""Extract simple scene graphs from LIBERO observations and language instructions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np


IN_XY_THRESHOLD_DEFAULT = 0.09
IN_XY_THRESHOLD_BASKET = 0.11
IN_Z_MIN = -0.03
IN_Z_MAX = 0.16
HOLDING_DIST_THRESHOLD = 0.05
HOLDING_RELATIVE_DIST_THRESHOLD = 0.08
HOLDING_HEIGHT_THRESHOLD = 0.03
HOLDING_CONTACT_REL_THRESHOLD = 0.025
HOLDING_CONTACT_HEIGHT_THRESHOLD = 0.02
GRIPPER_ENGAGED_QPOS_THRESHOLD = 0.035

OBJECT_ONTOLOGY = {
    'cream_cheese_1': {
        'holding_rel_dist_threshold': 0.14,
        'holding_quat_similarity_threshold': 0.82,
        'holding_contact_rel_threshold': 0.06,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [-0.07, -0.99, -0.03, 0.07],
    },
    'butter_1': {
        'holding_rel_dist_threshold': 0.16,
        'holding_quat_similarity_threshold': 0.76,
        'holding_contact_rel_threshold': 0.07,
        'holding_contact_height_threshold': 0.06,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'milk_1': {
        'holding_rel_dist_threshold': 0.12,
        'holding_quat_similarity_threshold': 0.70,
        'holding_contact_rel_threshold': 0.055,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'salad_dressing_1': {
        'holding_rel_dist_threshold': 0.12,
        'holding_quat_similarity_threshold': 0.70,
        'holding_contact_rel_threshold': 0.055,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'tomato_sauce_1': {
        'holding_rel_dist_threshold': 0.12,
        'holding_quat_similarity_threshold': 0.70,
        'holding_contact_rel_threshold': 0.055,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'ketchup_1': {
        'holding_rel_dist_threshold': 0.12,
        'holding_quat_similarity_threshold': 0.70,
        'holding_contact_rel_threshold': 0.055,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'bbq_sauce_1': {
        'holding_rel_dist_threshold': 0.12,
        'holding_quat_similarity_threshold': 0.70,
        'holding_contact_rel_threshold': 0.055,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'orange_juice_1': {
        'holding_rel_dist_threshold': 0.12,
        'holding_quat_similarity_threshold': 0.70,
        'holding_contact_rel_threshold': 0.055,
        'holding_contact_height_threshold': 0.05,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'chocolate_pudding_1': {
        'holding_rel_dist_threshold': 0.15,
        'holding_quat_similarity_threshold': 0.78,
        'holding_contact_rel_threshold': 0.065,
        'holding_contact_height_threshold': 0.055,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
}

TASK_ONTOLOGY = {
    'pick up the alphabet soup and place it in the basket': {
        'primary_object': 'alphabet_soup_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the cream cheese and place it in the basket': {
        'primary_object': 'cream_cheese_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the salad dressing and place it in the basket': {
        'primary_object': 'salad_dressing_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the bbq sauce and place it in the basket': {
        'primary_object': 'bbq_sauce_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the ketchup and place it in the basket': {
        'primary_object': 'ketchup_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the tomato sauce and place it in the basket': {
        'primary_object': 'tomato_sauce_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the butter and place it in the basket': {
        'primary_object': 'butter_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the milk and place it in the basket': {
        'primary_object': 'milk_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the chocolate pudding and place it in the basket': {
        'primary_object': 'chocolate_pudding_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
    'pick up the orange juice and place it in the basket': {
        'primary_object': 'orange_juice_1',
        'target_container': 'basket_1',
        'container_xy_threshold': 0.14,
        'container_z_min': -0.05,
        'container_z_max': 0.20,
        'holding_rel_dist_threshold': 0.09,
        'holding_quat_similarity_threshold': 0.94,
        'holding_reference_quat': [0.0, -0.70, 0.71, 0.03],
    },
}


def _merge_object_ontology(task_ontology: Dict[str, Any], object_id: str) -> Dict[str, Any]:
    merged = dict(task_ontology)
    merged.update(OBJECT_ONTOLOGY.get(object_id, {}))
    return merged


def extract_scene_graph_from_libero_observation(obs: Dict[str, Any], task_description: str = "") -> Dict[str, Any]:
    object_ids = sorted({key[:-4] for key in obs.keys() if key.endswith('_pos') and not key.startswith('robot0_') and not key.endswith('_eef_pos')})
    ontology = TASK_ONTOLOGY.get(task_description.lower().strip(), {})

    eef_pos = np.asarray(obs.get('robot0_eef_pos', np.zeros(3)), dtype=float)
    gripper_qpos = np.asarray(obs.get('robot0_gripper_qpos', np.zeros(2)), dtype=float)
    gripper_engaged = bool(np.mean(np.abs(gripper_qpos)) >= GRIPPER_ENGAGED_QPOS_THRESHOLD)
    gripper_closed = gripper_engaged

    objects: List[Dict[str, Any]] = [
        {
            'id': 'robot_1',
            'label': 'robot',
            'class': 'robot',
            'pose': {'position': eef_pos.tolist()},
        },
        {
            'id': 'table_1',
            'label': 'table',
            'class': 'table',
        },
    ]
    relations: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []

    positions: Dict[str, np.ndarray] = {}
    for object_id in object_ids:
        pos = np.asarray(obs.get(f'{object_id}_pos'), dtype=float)
        quat = np.asarray(obs.get(f'{object_id}_quat', np.array([0.0, 0.0, 0.0, 1.0])), dtype=float)
        positions[object_id] = pos
        objects.append(
            {
                'id': object_id,
                'label': object_id.replace('_1', '').replace('_', ' '),
                'class': object_id.rsplit('_', 1)[0],
                'pose': {
                    'position': pos.tolist(),
                    'orientation_xyzw': quat.tolist(),
                },
            }
        )

    basket_ids = [object_id for object_id in object_ids if 'basket' in object_id or 'bowl' in object_id or 'tray' in object_id]
    if ontology.get('target_container') and ontology['target_container'] in object_ids and ontology['target_container'] not in basket_ids:
        basket_ids.append(ontology['target_container'])
    container_positions = {object_id: positions[object_id] for object_id in basket_ids}

    for object_id, pos in positions.items():
        if object_id in basket_ids:
            relations.append({'type': 'on', 'subject': object_id, 'object': 'table_1'})
            continue

        object_ontology = _merge_object_ontology(ontology, object_id)
        rel_pos = np.asarray(obs.get(f'{object_id}_to_robot0_eef_pos', np.full(3, np.inf)), dtype=float)
        rel_quat = np.asarray(obs.get(f'{object_id}_to_robot0_eef_quat', np.array([0.0, 0.0, 0.0, 1.0])), dtype=float)
        eef_dist = float(np.linalg.norm(pos - eef_pos))
        rel_dist = float(np.linalg.norm(rel_pos))
        reference_quat = np.asarray(object_ontology.get('holding_reference_quat', [0.0, 0.0, 0.0, 1.0]), dtype=float)
        quat_similarity = _quat_similarity(rel_quat, reference_quat)
        holding_rel_threshold = float(object_ontology.get('holding_rel_dist_threshold', HOLDING_RELATIVE_DIST_THRESHOLD))
        holding_quat_threshold = float(object_ontology.get('holding_quat_similarity_threshold', 0.55))
        holding_candidate = (
            gripper_closed
            and eef_dist < HOLDING_DIST_THRESHOLD
            and rel_dist < holding_rel_threshold
            and abs(rel_pos[2]) < HOLDING_HEIGHT_THRESHOLD
            and quat_similarity > holding_quat_threshold
        )
        tight_contact = (
            gripper_engaged
            and rel_dist < float(object_ontology.get('holding_contact_rel_threshold', HOLDING_CONTACT_REL_THRESHOLD))
            and abs(rel_pos[2]) < float(object_ontology.get('holding_contact_height_threshold', HOLDING_CONTACT_HEIGHT_THRESHOLD))
            and quat_similarity > max(0.70, holding_quat_threshold - 0.25)
        )
        if ontology.get('primary_object') == object_id or object_id in OBJECT_ONTOLOGY:
            if gripper_closed and rel_dist < holding_rel_threshold * 1.15:
                holding_candidate = holding_candidate or quat_similarity > max(0.45, holding_quat_threshold - 0.15)
            holding_candidate = holding_candidate or tight_contact
        if holding_candidate:
            relations.append({'type': 'holding', 'subject': 'robot_1', 'object': object_id})
            continue

        placed_in_container = False
        for container_id, container_pos in container_positions.items():
            xy_dist = np.linalg.norm(pos[:2] - container_pos[:2])
            z_delta = pos[2] - container_pos[2]
            xy_threshold = IN_XY_THRESHOLD_BASKET if 'basket' in container_id else IN_XY_THRESHOLD_DEFAULT
            z_min = IN_Z_MIN
            z_max = IN_Z_MAX
            if ontology.get('target_container') == container_id:
                xy_threshold = float(ontology.get('container_xy_threshold', xy_threshold + 0.01))
                z_min = float(ontology.get('container_z_min', z_min))
                z_max = float(ontology.get('container_z_max', z_max))
            if xy_dist < xy_threshold and z_min <= z_delta <= z_max:
                relations.append({'type': 'in', 'subject': object_id, 'object': container_id})
                placed_in_container = True
                break
        if placed_in_container:
            continue

        relations.append({'type': 'on', 'subject': object_id, 'object': 'table_1'})

    for object_id, pos in positions.items():
        for other_id, other_pos in positions.items():
            if object_id == other_id:
                continue
            delta = pos - other_pos
            xy_dist = np.linalg.norm(delta[:2])
            if xy_dist < 0.12:
                if delta[0] < -0.04:
                    relations.append({'type': 'left_of', 'subject': object_id, 'object': other_id})
                elif delta[0] > 0.04:
                    relations.append({'type': 'right_of', 'subject': object_id, 'object': other_id})
                if delta[1] < -0.04:
                    relations.append({'type': 'behind', 'subject': object_id, 'object': other_id})
                elif delta[1] > 0.04:
                    relations.append({'type': 'in_front_of', 'subject': object_id, 'object': other_id})

    if gripper_closed:
        states.append({'object': 'robot_1', 'state': 'gripper_closed'})
    else:
        states.append({'object': 'robot_1', 'state': 'gripper_open'})

    return {
        'objects': objects,
        'relations': _dedupe_relations(relations),
        'states': _dedupe_states(states),
        'source': 'libero_observation_extractor.v1',
        'task_description': task_description,
    }



def build_goal_scene_graph_from_instruction(task_description: str, current_scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    instruction = task_description.lower().strip()
    objects = list(current_scene_graph.get('objects', []))
    object_ids = {obj['id'] for obj in objects if isinstance(obj, dict) and 'id' in obj}
    ontology = TASK_ONTOLOGY.get(instruction, {})

    basket_id = ontology.get('target_container') or _match_object_id('basket', object_ids, default='basket_1')

    if _is_all_objects_to_container_instruction(instruction):
        movable_object_ids = _select_movable_object_ids(object_ids, current_scene_graph, basket_id)
        return {
            'objects': objects,
            'relations': [{'type': 'in', 'subject': object_id, 'object': basket_id} for object_id in movable_object_ids],
            'states': [],
            'source': 'instruction_goal_builder.v2',
        }

    multi_object_ids = _extract_multi_object_transport_ids(instruction, object_ids)
    if multi_object_ids and _targets_container_instruction(instruction):
        return {
            'objects': objects,
            'relations': [{'type': 'in', 'subject': object_id, 'object': basket_id} for object_id in multi_object_ids],
            'states': [],
            'source': 'instruction_goal_builder.v2',
        }

    if 'place it in the basket' in instruction and instruction.startswith('pick up the '):
        object_label = instruction.replace('pick up the ', '').replace(' and place it in the basket', '').strip()
        object_id = ontology.get('primary_object') or _match_object_id(object_label, object_ids)
        return {
            'objects': objects,
            'relations': [{'type': 'in', 'subject': object_id, 'object': basket_id}],
            'states': [],
            'source': 'instruction_goal_builder.v2',
        }

    if 'place it on the ' in instruction and instruction.startswith('pick up the '):
        prefix, target_label = instruction.split(' and place it on the ', 1)
        object_label = prefix.replace('pick up the ', '').strip()
        object_id = _match_object_id(object_label, object_ids)
        target_id = _match_object_id(target_label.strip(), object_ids)
        return {
            'objects': objects,
            'relations': [{'type': 'on', 'subject': object_id, 'object': target_id}],
            'states': [],
            'source': 'instruction_goal_builder.v2',
        }

    if instruction.startswith('open the '):
        object_id = _match_object_id(instruction.replace('open the ', '').strip(), object_ids)
        return {'objects': objects, 'relations': [], 'states': [{'object': object_id, 'state': 'open'}], 'source': 'instruction_goal_builder.v2'}

    if instruction.startswith('close the '):
        object_id = _match_object_id(instruction.replace('close the ', '').strip(), object_ids)
        return {'objects': objects, 'relations': [], 'states': [{'object': object_id, 'state': 'closed'}], 'source': 'instruction_goal_builder.v2'}

    return {'objects': objects, 'relations': [], 'states': [], 'source': 'instruction_goal_builder.v2'}


def _is_all_objects_to_container_instruction(instruction: str) -> bool:
    return (
        ('all objects' in instruction or 'all the objects' in instruction or 'everything' in instruction)
        and _targets_container_instruction(instruction)
    )


def _targets_container_instruction(instruction: str) -> bool:
    return any(token in instruction for token in ('basket', 'bowl', 'tray', 'container'))


def _select_movable_object_ids(object_ids, current_scene_graph: Dict[str, Any], container_id: str | None) -> List[str]:
    excluded_prefixes = ('robot', 'table')
    excluded_keywords = ('basket', 'bowl', 'tray', 'drawer', 'microwave', 'cabinet', 'plate')
    already_inside = {
        relation.get('subject')
        for relation in current_scene_graph.get('relations', [])
        if relation.get('type') == 'in' and relation.get('object') == container_id
    }
    movable = []
    for object_id in sorted(object_ids):
        if not object_id or object_id in already_inside:
            continue
        if object_id.startswith(excluded_prefixes):
            continue
        if any(keyword in object_id for keyword in excluded_keywords):
            continue
        movable.append(object_id)
    return movable


def _extract_multi_object_transport_ids(instruction: str, object_ids) -> List[str]:
    if not _targets_container_instruction(instruction):
        return []

    text = instruction
    for separator in ('put ', 'place ', 'move ', 'take ', 'pick up '):
        if text.startswith(separator):
            text = text[len(separator):]
            break
    for suffix in (' in the basket', ' into the basket', ' in the bowl', ' into the bowl', ' in the tray', ' into the tray'):
        if suffix in text:
            text = text.split(suffix, 1)[0]
            break
    text = text.replace(' and then ', ', ')
    text = text.replace(' and ', ', ')
    candidates = [part.strip() for part in text.split(',') if part.strip()]

    matched = []
    seen = set()
    for candidate in candidates:
        candidate = re.sub(r'^(the|a|an)\s+', '', candidate).strip()
        object_id = _match_object_id(candidate, object_ids, default=None)
        if object_id and object_id not in seen:
            matched.append(object_id)
            seen.add(object_id)
    return matched



def _match_object_id(label: str, object_ids, default: str | None = None) -> str:
    normalized_label = _normalize_text(label)
    for object_id in object_ids:
        if normalized_label and normalized_label in _normalize_text(object_id):
            return object_id
    if default is not None:
        return default
    raise ValueError(f'Could not match object label {label!r} to known object ids: {sorted(object_ids)}')



def _normalize_text(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def _quat_similarity(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    quat_a = np.asarray(quat_a, dtype=float)
    quat_b = np.asarray(quat_b, dtype=float)
    if np.linalg.norm(quat_a) == 0 or np.linalg.norm(quat_b) == 0:
        return 0.0
    quat_a = quat_a / np.linalg.norm(quat_a)
    quat_b = quat_b / np.linalg.norm(quat_b)
    return float(abs(np.dot(quat_a, quat_b)))



def _dedupe_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for rel in relations:
        key = (rel.get('type'), rel.get('subject'), rel.get('object'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped



def _dedupe_states(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for state in states:
        key = (state.get('object'), state.get('state'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(state)
    return deduped
