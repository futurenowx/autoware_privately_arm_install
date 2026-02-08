# AutoSteer + EgoLanes (Video Input, Headless-Safe, Fast Preview, Safe Checkpoints)
import sys
import cv2
import numpy as np
import os
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime
import subprocess
from tqdm import tqdm
import torch
from torchvision import transforms

sys.path.append('../..')
from Models.inference.auto_steer_infer import AutoSpeedNetworkInfer
from inference.ego_lanes_infer import EgoLanesNetworkInfer

# -------------------------
# Frame sizes
# -------------------------
OUTPUT_SIZE = (1280, 720)
PREVIEW_SIZE = (720, 360)
INF_SIZE = (640, 320)


# -------------------------
# Utility functions
# -------------------------
def rotate_wheel(wheel_img, angle_deg):
    h, w = wheel_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(
        wheel_img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


def overlay_on_top(base_img, wheel_img, frame_time, steering_angle):
    H, W = base_img.shape[:2]
    oh, ow = wheel_img.shape[:2]
    x, y = W - ow - 60, 20

    image = base_img.copy()
    alpha = wheel_img[:, :, 3] / 255.0
    for c in range(3):
        image[y:y+oh, x:x+ow, c] = (
            wheel_img[:, :, c] * alpha +
            image[y:y+oh, x:x+ow, c] * (1 - alpha)
        )

    cv2.putText(image, frame_time, (x - 60, y + oh + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, f"{steering_angle:.2f} deg", (x - 60, y + oh + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


def overlay_egolanes(frame, lane_mask, threshold=0.3):
    overlay = frame.copy()
    H, W = frame.shape[:2]

    if lane_mask.ndim == 3:
        colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (128, 0, 128)]
        for i in range(lane_mask.shape[0]):
            lm = cv2.resize(lane_mask[i], (W, H), cv2.INTER_NEAREST)
            overlay[lm >= threshold] = colors[i % len(colors)]
    else:
        lm = cv2.resize(lane_mask, (W, H), cv2.INTER_NEAREST)
        overlay[lm >= threshold] = (0, 255, 0)

    return cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)


# -------------------------
# Main
# -------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--egolanes_checkpoint_path", required=True)
    parser.add_argument("-a", "--autosteer_checkpoint_path", required=True)
    parser.add_argument("-i", "--input_video_path", required=True)
    parser.add_argument("-o", "--output_video_path", required=True)
    parser.add_argument("--show", action="store_true", help="Headless preview via ffplay")
    parser.add_argument("--max-fps", type=float, default=60.0, help="Maximum preview FPS")
    args = parser.parse_args()

    args.input_video_path = os.path.expanduser(args.input_video_path)
    args.output_video_path = os.path.expanduser(args.output_video_path)

    # -------------------------
    # Load models
    # -------------------------
    print("Loading models...")
    autosteer = AutoSpeedNetworkInfer(
        args.egolanes_checkpoint_path,
        args.autosteer_checkpoint_path
    )
    egolanes = EgoLanesNetworkInfer(args.egolanes_checkpoint_path)
    print("Models loaded")

    # -------------------------
    # Open video
    # -------------------------
    cap = cv2.VideoCapture(args.input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------------------------
    # Video writer
    # -------------------------
    os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output_video_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        OUTPUT_SIZE
    )

    # -------------------------
    # Load wheel image
    # -------------------------
    wheel_path = os.path.join(os.path.dirname(__file__), "../../../Media/wheel_green.png")
    wheel = cv2.imread(wheel_path, cv2.IMREAD_UNCHANGED)
    if wheel is None:
        raise FileNotFoundError(f"Wheel image not found: {wheel_path}")
    wheel = cv2.resize(wheel, None, fx=0.8, fy=0.8)

    # -------------------------
    # Launch ffplay preview if requested
    # -------------------------
    ffplay = None
    if args.show:
        ffplay = subprocess.Popen(
            ["ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
             "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}", "-i", "pipe:0", "-loglevel", "quiet"],
            stdin=subprocess.PIPE
        )

    prev_time = 0.0
    print(f"Processing {total_frames} frames...")

    # -------------------------
    # Process frames
    # -------------------------
    for _ in tqdm(range(total_frames), desc="Processing", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(INF_SIZE)

        # -------------------------
        # Model inference
        # -------------------------
        angle = autosteer.inference(pil)
        lanes = egolanes.inference(np.array(pil))

        # -------------------------
        # Visualization
        # -------------------------
        frame_vis = cv2.resize(frame, OUTPUT_SIZE)
        frame_vis = overlay_egolanes(frame_vis, lanes)
        frame_vis = overlay_on_top(
            frame_vis,
            rotate_wheel(wheel, angle),
            datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
            angle
        )

        # -------------------------
        # Write to output video
        # -------------------------
        writer.write(frame_vis)

        # -------------------------
        # Fast headless preview
        # -------------------------
        if args.show and ffplay and ffplay.stdin:
            now = cv2.getTickCount() / cv2.getTickFrequency()
            if now - prev_time >= 1.0 / args.max_fps:
                try:
                    preview_frame = cv2.resize(frame_vis, PREVIEW_SIZE)
                    ffplay.stdin.write(preview_frame.tobytes())
                    ffplay.stdin.flush()
                    prev_time = now
                except BrokenPipeError:
                    pass

    # -------------------------
    # Cleanup
    # -------------------------
    cap.release()
    writer.release()

    if ffplay:
        if ffplay.poll() is None:
            ffplay.stdin.close()
            ffplay.terminate()
            ffplay.wait()

    print(f"Visualization video saved to: {args.output_video_path}")


if __name__ == "__main__":
    main()

