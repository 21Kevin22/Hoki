import numpy as np

from occ_vla.world_model.arm_free_subgoal import ARM_OCC_THRESHOLD, TOKEN_GRID_SIDE, ArmFreeSubgoalGenerator


def test_should_trigger_above_threshold():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    assert gen.should_trigger(ARM_OCC_THRESHOLD + 0.01)
    assert not gen.should_trigger(ARM_OCC_THRESHOLD - 0.01)


def test_arm_token_mask_all_arm():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.ones((512, 512), dtype=bool)
    token_mask = gen._arm_token_mask(mask)  # noqa: SLF001
    assert token_mask.shape == (TOKEN_GRID_SIDE**2,)
    assert token_mask.all()


def test_arm_token_mask_no_arm():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    token_mask = gen._arm_token_mask(mask)  # noqa: SLF001
    assert not token_mask.any()


def test_arm_token_mask_partial():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    mask[:16, :16] = True  # confined to the first token's receptive field
    token_mask = gen._arm_token_mask(mask).reshape(TOKEN_GRID_SIDE, TOKEN_GRID_SIDE)  # noqa: SLF001
    assert token_mask[0, 0]
    assert not token_mask[0, 1]
    assert not token_mask[1, 0]


def test_gripper_end_bbox_token_mask_covers_only_lowest_rows():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    # diagonal-ish arm silhouette spanning the whole frame vertically
    mask[0:16, 0:32] = True  # mount, top rows
    mask[480:512, 480:512] = True  # gripper end, bottom rows
    token_mask = gen._gripper_end_bbox_token_mask(mask, row_fraction=0.35, pad_tokens=0).reshape(  # noqa: SLF001
        TOKEN_GRID_SIDE, TOKEN_GRID_SIDE
    )
    # top-left (mount) tokens must NOT be masked -- only the gripper end should be
    assert not token_mask[0, 0]
    # bottom-right (gripper end) tokens must be masked
    assert token_mask[-1, -1]


def test_gripper_end_bbox_token_mask_no_arm():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    token_mask = gen._gripper_end_bbox_token_mask(mask)  # noqa: SLF001
    assert not token_mask.any()


def test_gripper_end_area_token_mask_hits_target_area_approximately():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    mask[0:16, 0:32] = True  # mount, top rows -- shouldn't affect area, only anchors off the lowest row
    mask[400:432, 200:232] = True  # gripper end, away from any image edge

    for target_fraction in (0.05, 0.10, 0.15, 0.20):
        token_mask = gen._gripper_end_area_token_mask(mask, target_area_fraction=target_fraction)  # noqa: SLF001
        actual_fraction = token_mask.sum() / (TOKEN_GRID_SIDE**2)
        # side_tokens is a rounded integer sqrt, so exact area isn't hit,
        # but should be close for an anchor far from any grid edge
        assert abs(actual_fraction - target_fraction) < 0.02


def test_gripper_end_area_token_mask_anchored_at_lowest_row():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    mask[0:16, 0:32] = True
    mask[480:512, 240:272] = True  # gripper end near the bottom edge

    token_mask = gen._gripper_end_area_token_mask(mask, target_area_fraction=0.10).reshape(  # noqa: SLF001
        TOKEN_GRID_SIDE, TOKEN_GRID_SIDE
    )
    # the region should include the bottom row (gripper end), not the top (mount)
    assert token_mask[-1, :].any()
    assert not token_mask[0, :].any()


def test_gripper_end_area_token_mask_no_arm():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    token_mask = gen._gripper_end_area_token_mask(mask, target_area_fraction=0.10)  # noqa: SLF001
    assert not token_mask.any()


def test_gripper_end_area_token_mask_clips_near_edge():
    gen = ArmFreeSubgoalGenerator(world_model=None)
    mask = np.zeros((512, 512), dtype=bool)
    mask[496:512, 0:16] = True  # gripper end in the bottom-left corner
    # must not raise, and must stay within the grid
    token_mask = gen._gripper_end_area_token_mask(mask, target_area_fraction=0.20).reshape(  # noqa: SLF001
        TOKEN_GRID_SIDE, TOKEN_GRID_SIDE
    )
    assert token_mask.shape == (TOKEN_GRID_SIDE, TOKEN_GRID_SIDE)
    assert token_mask[-1, 0]
