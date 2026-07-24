"""Runs in the `yoloworld` conda env. Serves open-vocabulary detection
requests over the same file-based RPC used by pi05_worker/sd_worker,
so the LIBERO rollout loop (base env, incompatible with ultralytics'
own torch/numpy pins) can call it without merging environments.

2026-07-23 finding (see CLAUDE.md-equivalent session log): YOLO-World
(yolov8s-worldv2, CLIP-based open-vocab) trained on real photos shows a
real domain gap on LIBERO's flat-shaded synthetic renders -- "moka pot"
from an unusual top-down angle got ~0.01 confidence (unusable), but a
common category viewed normally ("mug") got ~0.31 (usable, if not
great). This worker does NOT paper over that; it returns whatever the
model gives, including low-confidence/empty results, so the caller can
make its own confidence-threshold decision (matching the project's
established last_known_position "hold last confident position, don't
trust noise" pattern) rather than have the worker silently decide.

Run:
  conda activate yoloworld
  CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/_workers/yolo_world_worker.py
"""

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rpc  # noqa: E402

RPC_DIR = os.environ.get("YOLO_WORLD_WORKER_RPC_DIR", str(_ROOT / ".rpc" / "yolo_world"))
MODEL_NAME = "yolov8s-worldv2.pt"


def main():
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(MODEL_NAME)
    model.to("cuda:0")
    print(f"[yolo_world worker] loaded, serving {RPC_DIR}", flush=True)

    def handler(arrays, fields):
        image = arrays["image"]  # HWC uint8
        class_names = fields["class_names"].split("|")
        # 2026-07-23: calling set_classes() a second time (different
        # vocabulary) after the model was already used once crashed
        # with a cuda:0/cpu device mismatch inside CLIP's text
        # embedding -- set_classes' text encoder pass doesn't reliably
        # stay on the model's device across repeated calls. Re-assert
        # placement every call; cheap relative to inference itself.
        model.to("cuda:0")
        model.set_classes(class_names)
        model.to("cuda:0")
        results = model.predict(image, conf=0.005, verbose=False, device=0)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"boxes": np.zeros((0, 4), dtype=np.float32), "confs": np.zeros((0,), dtype=np.float32)}, \
                   {"class_names_out": ""}
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = r.boxes.conf.cpu().numpy().astype(np.float32)
        classes = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
        order = np.argsort(-confs)
        boxes, confs = boxes[order], confs[order]
        classes = [classes[i] for i in order]
        return {"boxes": boxes, "confs": confs}, {"class_names_out": "|".join(classes)}

    rpc.serve(RPC_DIR, handler)


if __name__ == "__main__":
    main()
