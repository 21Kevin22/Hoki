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
