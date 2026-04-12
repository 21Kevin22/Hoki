import os
import time
from dataclasses import dataclass

import cv2
import draccus
import json_numpy
import numpy as np
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import prepare_observation, process_action

json_numpy.patch()


@dataclass
class SceneGraphRolloutConfig:
    task_suite_name: str = "libero_90"
    task_id: int = 0
    episode_idx: int = 0
    horizon: int = 5000
    env_img_res: int = 256
    # 多くのLIBERO環境では +1.0 が開く / -1.0 が閉じる
    gripper_open_cmd: float = 1.0
    gripper_close_cmd: float = -1.0
    # カンマ区切りで複数指定 (例: "tomato_sauce_1,tomato_sauce_2")
    object_ids_csv: str = "tomato_sauce_1"
    basket_id: str = "basket_1"
    # カゴ中心へ寄せるXYオフセット（m単位）
    basket_xy_offset_x: float = 0.0
    basket_xy_offset_y: float = 0.0
    # 1つの物体を掴む最大試行回数（失敗したら次の物体へ）
    per_object_attempts: int = 3


def execute_step(env, name, loops, target_pos, grip, tol, replay_images, cur_obs):
    """指定された目標位置とグリッパー状態を保持して実行する。"""
    for _ in range(loops):
        if cur_obs is None:
            return None, False

        eef_p = np.asarray(cur_obs.get("robot0_eef_pos", [0.0, 0.0, 0.0]))
        d_pos = target_pos - eef_p

        dz = d_pos[2] * 6.0
        if "LOWER" in name and d_pos[2] < 0:
            dz = min(dz, -0.06)

        action = np.array(
            [
                np.clip(d_pos[0] * 5.0, -0.12, 0.12),
                np.clip(d_pos[1] * 5.0, -0.12, 0.12),
                np.clip(dz, -0.8, 0.8),
                0.0,
                0.0,
                0.0,
                grip,
            ]
        )

        cur_obs, _, done, _ = env.step(process_action(action, "openvla").tolist())
        if cur_obs:
            _, img = prepare_observation(cur_obs, 224)
            replay_images.append(img)

        if tol > 0 and np.linalg.norm(d_pos) <= tol and "STAY" not in name:
            break
        if done:
            break

    return cur_obs, True


def run_full_sequence(env, obs, replay_images, obj_id, basket_id, open_cmd, close_cmd, basket_xy_offset):
    """移動→開く→下降→把持→持ち上げ→搬送→リリースの順で実行する。"""
    cur_obs = obs
    obj_p = np.asarray(cur_obs.get(f"{obj_id}_pos", [0.0, 0.0, 0.0]))
    basket_p = np.asarray(cur_obs.get(f"{basket_id}_pos", [0.0, 0.0, 0.0]))
    basket_p = basket_p + np.array([basket_xy_offset[0], basket_xy_offset[1], 0.0])

    # 1) 物体の真上まで移動（最初から開いた状態で移動）
    cur_obs, _ = execute_step(
        env,
        "MOVE_ABOVE_OPEN",
        60,
        np.array([obj_p[0], obj_p[1], obj_p[2] + 0.15]),
        open_cmd,
        0.02,
        replay_images,
        cur_obs,
    )

    # 2) 真上で明示的に開く（保持時間を長めにして確実に開く）
    cur_obs, _ = execute_step(
        env,
        "STAY_OPEN",
        40,
        np.array([obj_p[0], obj_p[1], obj_p[2] + 0.15]),
        open_cmd,
        -1,
        replay_images,
        cur_obs,
    )

    # 3) 開いたまま物体高さまで下降
    cur_obs, _ = execute_step(
        env,
        "LOWER_TO_OBJ_OPEN",
        90,
        np.array([obj_p[0], obj_p[1], obj_p[2] + 0.005]),
        open_cmd,
        0.002,
        replay_images,
        cur_obs,
    )

    # 4) その場で閉じる
    cur_obs, _ = execute_step(
        env,
        "STAY_CLOSE",
        60,
        np.array([obj_p[0], obj_p[1], obj_p[2] + 0.005]),
        close_cmd,
        -1,
        replay_images,
        cur_obs,
    )

    # 5) 持ち上げ
    cur_obs, _ = execute_step(
        env,
        "LIFT_UP",
        60,
        np.array([obj_p[0], obj_p[1], obj_p[2] + 0.20]),
        close_cmd,
        0.05,
        replay_images,
        cur_obs,
    )

    eef_z = cur_obs.get("robot0_eef_pos", [0, 0, 0])[2]
    if eef_z < obj_p[2] + 0.10:
        print("    [!] Failed to lift object.")
        return cur_obs, False

    # 6) カゴへ搬送
    cur_obs, _ = execute_step(
        env,
        "MOVE_TO_BASKET",
        80,
        np.array([basket_p[0], basket_p[1], basket_p[2] + 0.20]),
        close_cmd,
        0.05,
        replay_images,
        cur_obs,
    )
    cur_obs, _ = execute_step(
        env,
        "LOWER_TO_BASKET",
        40,
        np.array([basket_p[0], basket_p[1], basket_p[2] + 0.10]),
        close_cmd,
        0.05,
        replay_images,
        cur_obs,
    )
    cur_obs, _ = execute_step(
        env,
        "RELEASE",
        40,
        np.array([basket_p[0], basket_p[1], basket_p[2] + 0.10]),
        open_cmd,
        -1,
        replay_images,
        cur_obs,
    )

    return cur_obs, True


def run_single_round(env, init_state, cfg: SceneGraphRolloutConfig):
    replay_images = []
    total_success = False

    object_ids = [s.strip() for s in cfg.object_ids_csv.split(",") if s.strip()]
    if not object_ids:
        raise ValueError("object_ids_csv is empty. Please specify at least one object id.")

    for trial in range(3):
        print(f"\n===== STARTING TRIAL {trial + 1} / 3 =====")
        obs = env.reset()
        obs = env.set_init_state(init_state)

        all_success = True
        for obj_id in object_ids:
            obj_success = False
            for attempt in range(cfg.per_object_attempts):
                obs, success = run_full_sequence(
                    env,
                    obs,
                    replay_images,
                    obj_id,
                    cfg.basket_id,
                    cfg.gripper_open_cmd,
                    cfg.gripper_close_cmd,
                    (cfg.basket_xy_offset_x, cfg.basket_xy_offset_y),
                )
                if success:
                    obj_success = True
                    break
                print(f"❌ Object {obj_id} attempt {attempt + 1}/{cfg.per_object_attempts} failed.")

            if not obj_success:
                all_success = False
                print(f"⚠️ Skipping object after {cfg.per_object_attempts} failed attempts: {obj_id}")
                continue

        if all_success:
            print(f"🎉 MISSION COMPLETE in Trial {trial + 1}!")
            total_success = True
            break
        print(f"❌ Trial {trial + 1} completed with skipped objects. Retrying...")

    save_dir = "/home/ubuntu/slocal/DiscreteDiffusionVLA/rollouts"
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"PICK_PLACE_FINAL_{int(time.time())}.mp4")

    if replay_images:
        h, w, _ = replay_images[0].shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for frame in replay_images:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"\n🎬 FULL VIDEO SAVED AT: {video_path} ({len(replay_images)} frames)")

    return total_success


@draccus.wrap()
def main(cfg: SceneGraphRolloutConfig):
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[cfg.task_suite_name]()
    task = suite.get_task(cfg.task_id)
    init_state = suite.get_task_init_states(cfg.task_id)[cfg.episode_idx]
    env, _ = get_libero_env(task, "openvla", resolution=cfg.env_img_res)

    def set_horizon(e, v):
        if hasattr(e, "horizon"):
            e.horizon = v
            return True
        return set_horizon(e.env, v) if hasattr(e, "env") else False

    set_horizon(env, cfg.horizon)

    try:
        run_single_round(env, init_state, cfg)
    finally:
        env.close()


if __name__ == "__main__":
    main()