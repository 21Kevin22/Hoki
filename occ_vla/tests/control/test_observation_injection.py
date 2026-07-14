import numpy as np

from occ_vla.control.observation_injection import OccVlaLiberoInputs


def _base_data(**extra):
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8),
        "prompt": "pick up the pot",
        **extra,
    }


def test_no_subgoal_zero_fills_right_wrist_like_libero_inputs():
    out = OccVlaLiberoInputs()(_base_data())
    assert out["image_mask"]["right_wrist_0_rgb"] == np.False_
    np.testing.assert_array_equal(out["image"]["right_wrist_0_rgb"], np.zeros((224, 224, 3), dtype=np.uint8))


def test_subgoal_image_fills_right_wrist_and_unmasks_it():
    subgoal = np.full((224, 224, 3), 7, dtype=np.uint8)
    out = OccVlaLiberoInputs()(_base_data(**{"observation/subgoal_image": subgoal}))
    assert out["image_mask"]["right_wrist_0_rgb"] == np.True_
    np.testing.assert_array_equal(out["image"]["right_wrist_0_rgb"], subgoal)


def test_cot_anchor_appended_to_prompt():
    out = OccVlaLiberoInputs()(_base_data(cot_anchor="the handle should now be under the gripper"))
    assert out["prompt"] == "pick up the pot the handle should now be under the gripper"


def test_is_pi0_fast_unmasks_right_wrist_even_without_subgoal():
    out = OccVlaLiberoInputs(is_pi0_fast=True)(_base_data())
    assert out["image_mask"]["right_wrist_0_rgb"] == np.True_
