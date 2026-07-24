"""Wiring for `OccVlaLiberoInputs` (observation_injection.py) into an
actual loadable policy -- option (a) from that module's docstring: a
new TrainConfig, built in occ_vla rather than by editing the vendored
third_party/openpi/src/openpi/training/config.py.

`LeRobotLiberoDataConfig.create()` (openpi's "pi05_libero" data config)
hardcodes `libero_policy.LiberoInputs(...)` as the one and only input
transform -- there's no hook to swap it via `repack_transforms` (those
run *before* it and would just get stomped, per
observation_injection.py's docstring). So `OccVlaLeRobotLiberoDataConfig`
below is a copy of that `create()` method with exactly one line changed:
`OccVlaLiberoInputs` in place of `LiberoInputs`. Everything else
(repack_transform, LiberoOutputs, model_transforms, extra_delta_transform
handling, asset/norm-stats loading via `create_base_config`) is
untouched, so a policy built this way is numerically identical to the
stock "pi05_libero" policy on any step where `subgoal_image`/`cot_anchor`
are absent -- this is an additive capability, not a behavior change to
existing experiments (all of which used the stock path and are
unaffected).
"""

import dataclasses

from openpi.models import model as _model
from openpi.training import config as _config

from occ_vla.control.observation_injection import OccVlaLiberoInputs


@dataclasses.dataclass(frozen=True)
class OccVlaLeRobotLiberoDataConfig(_config.LeRobotLiberoDataConfig):
    @property
    def _is_copy_of(self) -> str:
        # Documents the vendored source this was copied from, so a
        # future openpi upgrade that changes LeRobotLiberoDataConfig.create()
        # is something a reader would know to re-diff against.
        return "openpi.training.config.LeRobotLiberoDataConfig.create (as of the pi05_libero setup used in this project)"

    def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        from openpi import transforms as _transforms  # noqa: PLC0415
        from openpi.policies import libero_policy  # noqa: PLC0415

        # Unmodified from LeRobotLiberoDataConfig.create(): this
        # `repack_transform` is only consumed by openpi's *training* data
        # loader (to remap a LeRobot dataset's on-disk key names), not by
        # `create_trained_policy`'s inference path (which only uses the
        # `repack_transforms` function *argument*, left empty here) -- so
        # it never sees `subgoal_image`/`cot_anchor` and doesn't need to
        # know about them. (An earlier version of this file added those
        # keys here; `RepackTransform` does a strict `dict[key]` lookup
        # with no default, so that would have raised `KeyError` the
        # moment this data config was ever used for training, on every
        # example lacking a subgoal image. Left out on purpose.)
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[OccVlaLiberoInputs(is_pi0_fast=(model_config.model_type == _model.ModelType.PI0_FAST))],
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = _config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def build_occ_vla_train_config(base_config_name: str) -> _config.TrainConfig:
    """`base_config_name` is any existing pi0.5 config (e.g.
    "pi05_libero") -- returns a TrainConfig identical to it except
    `data` is swapped for `OccVlaLeRobotLiberoDataConfig`, carrying over
    the base config's own `repo_id`/`extra_delta_transform`. Pass the
    result straight to
    `openpi.policies.policy_config.create_trained_policy` (not through
    `openpi_config.get_config`, which only knows the vendored configs
    registered in third_party/openpi/src/openpi/training/config.py)."""
    base = _config.get_config(base_config_name)
    if not isinstance(base.data, _config.LeRobotLiberoDataConfig):
        raise TypeError(
            f"build_occ_vla_train_config expects a LIBERO config (data: LeRobotLiberoDataConfig), "
            f"got {type(base.data).__name__} for {base_config_name!r}"
        )
    occ_vla_data = OccVlaLeRobotLiberoDataConfig(
        repo_id=base.data.repo_id,
        assets=base.data.assets,
        base_config=base.data.base_config,
        extra_delta_transform=base.data.extra_delta_transform,
    )
    return dataclasses.replace(base, data=occ_vla_data)
