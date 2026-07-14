#!/usr/bin/env python
"""Train/fine-tune the MMaDA-8B world model via masked token prediction."""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
