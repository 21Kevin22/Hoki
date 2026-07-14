#!/usr/bin/env python
"""Run the integrated control loop on a live/sim robot."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    raise NotImplementedError("build ControlLoopComponents from cfg and run ControlLoop.step in a loop")


if __name__ == "__main__":
    main()
