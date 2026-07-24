import numpy as np

from occ_vla.pklp.visual_overlay import draw_kinematic_overlay


def test_draw_kinematic_overlay_marks_predicted_point():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    overlaid = draw_kinematic_overlay(image, current_position=np.array([2.0, 2.0]), predicted_position=np.array([20.0, 20.0]))

    assert overlaid.shape == image.shape
    assert overlaid[20, 20].tolist() == [255, 0, 0]
    # original untouched (returns a copy)
    assert image.max() == 0


def test_draw_kinematic_overlay_clips_out_of_bounds_prediction():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    # way outside the frame -- must not raise or wrap
    overlaid = draw_kinematic_overlay(image, current_position=np.array([0.0, 0.0]), predicted_position=np.array([500.0, -500.0]))

    assert overlaid.shape == image.shape
    # x=500 clips to w-1=15, y=-500 clips to 0 -- image[y, x]
    assert overlaid[0, 15].tolist() == [255, 0, 0]
