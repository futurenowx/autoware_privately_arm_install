#%%
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import subprocess
import torch
torch.cuda.empty_cache()
import time
from tqdm import tqdm
import cmapy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_3d_infer import Scene3DNetworkInfer

# -------------------------
OUTPUT_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 320)
INF_SIZE = (640, 320)
ALPHA_BLEND = 0.97
# -------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("-i", "--video_filepath", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("--skip_frames", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="Show live preview using ffplay")
    parser.add_argument("--max_fps", type=float, default=30.0, help="Maximum preview FPS")
    args = parser.parse_args()

    # Expand ~ and ensure output directory exists
    args.video_filepath = os.path.expanduser(args.video_filepath)
    args.output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Load Scene3D model
    print("Loading Scene3D model...")
    model = Scene3DNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("Scene3D Model Loaded")

    # Open video file
    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {args.video_filepath}")

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = args.skip_frames + 1
    fps_output = max(fps_input / skip, 1)

    # Setup video writer
    output_filepath_obj = args.output_file + ".avi"
    writer_obj = cv2.VideoWriter(
        output_filepath_obj,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps_output,
        OUTPUT_SIZE
    )
    if not writer_obj.isOpened():
        raise RuntimeError(f"Cannot open video writer at {output_filepath_obj}")
    print(f"Recording output to: {output_filepath_obj}")

    # Setup ffplay preview
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

    # Initialize frame counter
    frame_idx = 0
    print("Processing started...")

    # Use dynamic tqdm (do not rely on frame count from OpenCV)
    with tqdm(total=None, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break

            # Skip frames if needed
            if frame_idx % skip == 0:
                try:
                    # Convert frame for model
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb).resize(INF_SIZE)

                    # Run inference
                    with torch.no_grad():
                        prediction = model.inference(image_pil)

                    # Resize to original frame size
                    prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

                    # Normalize safely
                    denom = max(np.max(prediction) - np.min(prediction), 1e-5)
                    prediction_image = ((prediction - np.min(prediction)) * 255.0 / denom).astype(np.uint8)

                    # Apply color map
                    vis_obj = cv2.applyColorMap(prediction_image, cmapy.cmap("viridis"))

                    # Resize and blend with original frame
                    frame_hd = cv2.resize(frame, OUTPUT_SIZE)
                    vis_obj_hd = cv2.resize(vis_obj, OUTPUT_SIZE)
                    blended = cv2.addWeighted(vis_obj_hd, ALPHA_BLEND, frame_hd, 1 - ALPHA_BLEND, 0)

                    # Write to output
                    writer_obj.write(blended)

                    # Preview via ffplay
                    if ffplay:
                        now = time.time()
                        if now - prev_time >= 1.0 / args.max_fps:
                            preview_frame = cv2.resize(blended, PREVIEW_SIZE)
                            ffplay.stdin.write(preview_frame.tobytes())
                            ffplay.stdin.flush()
                            prev_time = now

                    pbar.update(1)

                except Exception as e:
                    print(f"Frame processing failed: {e}")

            frame_idx += 1

    # Cleanup
    cap.release()
    writer_obj.release()
    if ffplay:
        ffplay.stdin.close()
        ffplay.wait()

    print("Processing completed. Preview closed automatically.")

if __name__ == "__main__":
    main()
#%%

