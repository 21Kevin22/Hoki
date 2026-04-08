import cv2
import os
import numpy as np


class VideoRecorder:
    def __init__(self, output_path="outputs/run.mp4", fps=20):
        self.frames = []
        self.output_path = output_path
        self.fps = fps
        self.frame_count = 0

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def capture(self, frame):
        if frame is None:
            return

        frame = np.asarray(frame)
        if frame.ndim != 3:
            return
        if frame.shape[2] < 3:
            return

        frame = frame[:, :, :3]

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                max_value = float(frame.max()) if frame.size else 0.0
                if max_value <= 1.0:
                    frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        frame = np.ascontiguousarray(frame[:, :, ::-1])
        self.frames.append(frame)
        self.frame_count += 1

    def save(self):
        if len(self.frames) == 0:
            print("[VideoRecorder] no frames to save")
            return

        h, w, _ = self.frames[0].shape

        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h)
        )

        for f in self.frames:
            # OpenCVはBGR想定。RGBなら変換してもよい
            writer.write(f)

        writer.release()
        print(f"[VideoRecorder] saved {self.frame_count} frames: {self.output_path}")