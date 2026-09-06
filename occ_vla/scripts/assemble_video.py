"""occ_vla addition (2026-08-23): assembles frame_NNNNN.png sequences
saved by run_libero_occluded_oracle_headroom.py's --record-video-dir
into real .mp4 files, via cv2.VideoWriter (system ffmpeg is not
installed in this environment, but opencv-python ships its own mp4
encoder backend so no extra dependency is needed).

Usage: walks --root, treats every immediate subdirectory containing
frame_*.png as one episode, writes <subdir>.mp4 next to --root (or into
--out-dir if given).
"""
import argparse
import glob
import os

import cv2


def assemble_one(frame_dir, out_path, fps=15, suffix_glob="frame_[0-9][0-9][0-9][0-9][0-9].png"):
    # occ_vla fix (2026-09-06): the original "frame_*.png" glob also
    # matched frame_NNNNN_wrist.png (added later, 2026-08-30, for the
    # wrist-camera-bypass check) and, lexicographically, "00001." sorts
    # before "00001_", so the two cameras' frames interleaved into one
    # video instead of being written separately. Default pattern now
    # matches ONLY the plain 5-digit agentview frame; pass
    # suffix_glob="frame_[0-9][0-9][0-9][0-9][0-9]_wrist.png" explicitly
    # to assemble the wrist-camera video instead.
    frames = sorted(glob.glob(os.path.join(frame_dir, suffix_glob)))
    if not frames:
        return None
    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fp in frames:
        img = cv2.imread(fp)
        writer.write(img)
    writer.release()
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing one subdir per recorded episode")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    out_dir = args.out_dir or args.root
    os.makedirs(out_dir, exist_ok=True)

    for sub in sorted(os.listdir(args.root)):
        subpath = os.path.join(args.root, sub)
        if not os.path.isdir(subpath):
            continue
        out_path = os.path.join(out_dir, f"{sub}.mp4")
        n = assemble_one(subpath, out_path, fps=args.fps)
        if n:
            print(f"{sub}: {n} frames -> {out_path}")
        else:
            print(f"{sub}: no frames found, skipped")
        wrist_out_path = os.path.join(out_dir, f"{sub}_wrist.mp4")
        n_wrist = assemble_one(subpath, wrist_out_path, fps=args.fps,
                                suffix_glob="frame_[0-9][0-9][0-9][0-9][0-9]_wrist.png")
        if n_wrist:
            print(f"{sub} (wrist): {n_wrist} frames -> {wrist_out_path}")


if __name__ == "__main__":
    main()
