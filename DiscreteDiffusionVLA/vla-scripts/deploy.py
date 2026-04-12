"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from experiments.robot.scene_graph_planner import (
    plan_from_scene_graph_payload,
)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)

    def _decode_payload(self, payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        if "encoded" in payload:
            assert len(payload.keys()) == 1, "Only uses encoded payload!"
            return json.loads(payload["encoded"]), True
        return payload, False

    def _resolve_observation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        observation = payload.get("observation")
        if isinstance(observation, dict):
            return observation
        return payload

    def _predict_action(self, observation: Dict[str, Any], instruction: str):
        observation = dict(observation)
        observation["instruction"] = instruction
        return get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )

    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            payload, double_encode = self._decode_payload(payload)

            observation = self._resolve_observation(payload)
            instruction = observation["instruction"]

            action = self._predict_action(observation, instruction)

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def get_scene_graph_plan(self, payload: Dict[str, Any]) -> str:
        try:
            payload, double_encode = self._decode_payload(payload)
            plan = plan_from_scene_graph_payload(payload)
            if double_encode:
                return JSONResponse(json_numpy.dumps(plan))
            return JSONResponse(plan)
        except Exception:  # noqa: BLE001
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'instruction': str, 'current_scene_graph': dict, 'goal_scene_graph': dict}\n"
            )
            return "error"

    def get_scene_graph_plan_and_action(self, payload: Dict[str, Any]) -> str:
        try:
            payload, double_encode = self._decode_payload(payload)
            plan = plan_from_scene_graph_payload(payload)
            observation = self._resolve_observation(payload)
            next_action = plan.get("next_action")
            common_response = {
                "plan": plan,
                "subgoals": plan.get("subgoals", []),
                "num_subgoals": len(plan.get("subgoals", [])),
            }
            if next_action is None:
                response = {
                    **common_response,
                    "executed_instruction": None,
                    "planner_action": None,
                    "action": None,
                }
            else:
                instruction = next_action["instruction_text"]
                action = self._predict_action(observation, instruction)
                response = {
                    **common_response,
                    "executed_instruction": instruction,
                    "planner_action": next_action,
                    "action": action,
                }
            if double_encode:
                return JSONResponse(json_numpy.dumps(response))
            return JSONResponse(response)
        except Exception:  # noqa: BLE001
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'instruction': str, 'current_scene_graph': dict, 'goal_scene_graph': dict, 'observation': dict}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        self.app.post("/plan_from_scene_graph")(self.get_scene_graph_plan)
        self.app.post("/plan_and_act_from_scene_graph")(self.get_scene_graph_plan_and_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    offload_aux_modules_to_cpu: bool = False         # Keep action head / proprio projector on CPU to save VRAM

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
