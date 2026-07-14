from occ_vla.eval.metrics import Difficulty, SoccMetric


def test_difficulty_bands():
    metric = SoccMetric()
    assert metric.difficulty(0.0) == Difficulty.LIGHT
    assert metric.difficulty(0.29) == Difficulty.LIGHT
    assert metric.difficulty(0.3) == Difficulty.MEDIUM
    assert metric.difficulty(0.59) == Difficulty.MEDIUM
    assert metric.difficulty(0.6) == Difficulty.HEAVY
    assert metric.difficulty(0.99) == Difficulty.HEAVY
