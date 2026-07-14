#!/usr/bin/env python
"""PKLP itself is analytic (RAFT + constant-acceleration extrapolation),
not learned. This script fine-tunes the RAFT flow backbone on
in-domain robot manipulation video, if needed."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
