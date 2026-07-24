from occ_vla.world_model.mmada import _compounding_temperature_schedule


def test_compounding_temperature_schedule_matches_reference_recurrence():
    # Ground truth: hand-unroll third_party/mmada's own recurrence
    # (temperature = temperature * (1 - ratio), ratio = (i+1)/timesteps)
    # rather than re-deriving it -- this is a transcription check, not a
    # math re-derivation.
    timesteps = 18
    expected = []
    temperature = 1.0
    for step in range(timesteps):
        ratio = 1.0 * (step + 1) / timesteps
        temperature = temperature * (1.0 - ratio)
        expected.append(temperature)

    actual = _compounding_temperature_schedule(timesteps)

    assert actual == expected


def test_compounding_temperature_schedule_decays_faster_than_naive_noncompounding():
    # The bug this replaced: recomputing `initial_temperature * (1 - ratio)`
    # fresh each step instead of compounding. At step 8/18 the compounding
    # schedule should be well under half of the non-compounding value.
    timesteps = 18
    compounding = _compounding_temperature_schedule(timesteps)
    step = 8
    ratio = 1.0 * (step + 1) / timesteps
    noncompounding = 1.0 * (1.0 - ratio)

    assert compounding[step] < noncompounding * 0.5


def test_compounding_temperature_schedule_ends_at_zero():
    # ratio hits exactly 1.0 on the last step, zeroing the running product
    schedule = _compounding_temperature_schedule(10)
    assert schedule[-1] == 0.0
