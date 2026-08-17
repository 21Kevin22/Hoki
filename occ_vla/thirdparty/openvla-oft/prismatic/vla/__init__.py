# `materialize.py` (and its `.datasets`/`.datasets.rlds` chain) transitively
# imports `dlimp`/`tensorflow_datasets`/`tensorflow` -- needed only for
# RLDS TRAINING data loading, not for inference (occ_vla's eval scripts
# only ever need `prismatic.vla.action_tokenizer`/`.constants`, which don't
# depend on this chain). Made lazy (occ_vla local patch, 2026-08-18) because
# `import prismatic.models.vlas` (needed for inference too) transitively
# imports THIS package's `__init__.py`, so an eager import here forces the
# whole RLDS/tensorflow stack to load even for pure inference -- and on at
# least one real environment (Kaggle, this session) that stack's installed
# wheels had an unresolvable protobuf version conflict
# (tensorflow==2.15.0's own protobuf ceiling vs. tensorflow_metadata's
# generated code requiring `google.protobuf.runtime_version`, added only in
# protobuf>=5.26 -- no single protobuf version satisfies both).
# `get_vla_dataset_and_collator` itself is unused via this package-level
# re-export anywhere in this repo (grepped) -- training scripts import
# directly from `prismatic.vla.datasets`/`.materialize`, not through here.
try:
    from .materialize import get_vla_dataset_and_collator
except ImportError as _e:  # pragma: no cover - only triggered by a broken/absent RLDS training stack
    def get_vla_dataset_and_collator(*args, **kwargs):
        raise ImportError(
            "prismatic.vla.get_vla_dataset_and_collator needs the RLDS training "
            "stack (tensorflow/tensorflow_datasets/dlimp), which failed to import "
            f"in this environment: {_e}. Not needed for inference (occ_vla's own "
            "eval scripts never call this)."
        ) from _e
