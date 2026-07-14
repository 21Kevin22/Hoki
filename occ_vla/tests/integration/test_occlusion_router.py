from occ_vla.integration.occlusion_router import OcclusionRouter, OcclusionSignals, OcclusionSource


def test_routes_to_self_occlusion_above_threshold():
    router = OcclusionRouter(arm_occ_threshold=0.30)
    signals = OcclusionSignals(arm_s_occ=0.5, scene_dyn_occ=False)
    assert router.route(signals) == OcclusionSource.SELF


def test_routes_to_scene_when_not_self_occluded():
    router = OcclusionRouter(arm_occ_threshold=0.30)
    signals = OcclusionSignals(arm_s_occ=0.1, scene_dyn_occ=True)
    assert router.route(signals) == OcclusionSource.SCENE


def test_routes_to_none_when_clear():
    router = OcclusionRouter(arm_occ_threshold=0.30)
    signals = OcclusionSignals(arm_s_occ=0.0, scene_dyn_occ=False)
    assert router.route(signals) == OcclusionSource.NONE


def test_self_occlusion_takes_priority_over_scene():
    router = OcclusionRouter(arm_occ_threshold=0.30)
    signals = OcclusionSignals(arm_s_occ=0.5, scene_dyn_occ=True)
    assert router.route(signals) == OcclusionSource.SELF
