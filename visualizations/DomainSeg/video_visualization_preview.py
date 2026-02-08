#%%
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torch
import subprocess
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.domain_seg_infer import DomainSegNetworkInfer

# -------------------------
OUTPUT_SIZE = (1280, 720)  # optional output resizing
PREVIEW_SIZE = (640, 480)
INF_SIZE = (640, 320)
ALPHA_BLEND = 0.5
# -------------------------

def make_visualization(prediction):
    prediction = np.squeeze(prediction)
    row, col = prediction.shape
    vis = np.zeros((row, col, 3), dtype="uint8")
    # Background color
    vis[:, :, 0] = 255
    vis[:, :, 1] = 93
    vis[:, :, 2] = 61
    # Foreground color
    fg = np.where(prediction == 1.0)
    vis[fg[0], fg[1], 0] = 28
    vis[fg[0], fg[1], 1] = 148
    vis[fg[0], fg[1], 2] = 255
    return vis

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("-i", "--video_filepath", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("--skip_frames", type=int, default=0, help="Number of frames to skip")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--max_fps", type=float, default=30.0)
    args = parser.parse_args()

    # Expand paths and create output directory
    args.model_checkpoint_path = os.path.expanduser(args.model_checkpoint_path)
    args.video_filepath = os.path.expanduser(args.video_filepath)
    args.output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load model
    print("Loading DomainSeg model...")
    model = DomainSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("DomainSeg Model Loaded")

    # Open video
    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_filepath}")

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = args.skip_frames + 1
    fps_output = max(fps_input / skip, 1.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    writer_obj = cv2.VideoWriter(
        args.output_file + ".avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps_output,
        (frame_width, frame_height)
    )
    if not writer_obj.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.output_file}.avi")
    print(f"Recording output to {args.output_file}.avi")

    # FFplay preview setup
    ffplay = None
    prev_time = 0.0
    if args.show:
        ffplay = subprocess.Popen(
            [
                "ffplay",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}",
                "-framerate", str(int(args.max_fps)),
                "-autoexit",
                "-loglevel", "quiet",
                "-"
            ],
            stdin=subprocess.PIPE
        )

    # Total frames (used for progress bar)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = max(total_frames_in_video // skip, 1)

    print(f"Processing started on video file {args.video_filepath}...")
    total_frames = 0

    # Use tqdm progress bar
    with tqdm(total=frames_to_process, desc="Processing Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break

            total_frames += 1
            # Skip frames if requested
            if args.skip_frames > 0 and (total_frames - 1) % skip != 0:
                continue

            try:
                # Prepare frame for inference
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb).resize(INF_SIZE)

                # Model inference
                with torch.no_grad():
                    prediction = model.inference(image_pil)

                # Resize prediction to original frame size
                prediction = cv2.resize(np.squeeze(prediction), (frame.shape[1], frame.shape[0]))

                # Visualization
                vis_obj = make_visualization(prediction)
                blended = cv2.addWeighted(vis_obj, ALPHA_BLEND, frame, 1 - ALPHA_BLEND, 0)

                # Write output
                writer_obj.write(blended)

                # Preview via ffplay
                if ffplay:
                    now = time.time()
                    if now - prev_time >= 1.0 / args.max_fps:
                        preview_frame = cv2.resize(blended, PREVIEW_SIZE)
                        ffplay.stdin.write(preview_frame.tobytes())
                        prev_time = now

                pbar.update(1)

            except Exception as e:
                print(f"Frame processing failed: {e}")
                continue

    # Cleanup
    cap.release()
    writer_obj.release()
    if ffplay:
        ffplay.stdin.close()
        ffplay.wait()

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
#%%

