#!/usr/bin/env python
"""Evaluate the integrated system on the LIBERO-Occ benchmark, broken
down by Light/Medium/Heavy difficulty."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    raise NotImplementedError("instantiate LiberoOccEnv per task/difficulty and run rollouts")


if __name__ == "__main__":
    main()
