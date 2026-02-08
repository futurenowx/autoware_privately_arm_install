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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.domain_seg_infer import DomainSegNetworkInfer

# -------------------------
FRAME_INF_SIZE = (640, 320)  # model input size
FRAME_ORI_SIZE = (1280, 720)  # output video size
PREVIEW_SIZE = (640, 480)    # live preview size
ALPHA_BLEND = 0.5
# -------------------------

def make_visualization(prediction):
    prediction = np.squeeze(prediction)  # ensure 2D
    row, col = prediction.shape
    vis = np.zeros((row, col, 3), dtype="uint8")

    # Background color
    vis[:, :, 0] = 255
    vis[:, :, 1] = 93
    vis[:, :, 2] = 61

    # Foreground object color
    fg = np.where(prediction == 1.0)
    vis[fg[0], fg[1], 0] = 28
    vis[fg[0], fg[1], 1] = 148
    vis[fg[0], fg[1], 2] = 255

    return vis

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True, help="Path to DomainSeg PyTorch checkpoint")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device index")
    parser.add_argument("-o", "--output_video_path", default=None, help="Optional output video path")
    parser.add_argument("--show", action="store_true", help="Show live preview using ffplay")
    parser.add_argument("--max_fps", type=float, default=30.0, help="Maximum preview FPS")
    args = parser.parse_args()

    # Expand paths
    args.model_checkpoint_path = os.path.expanduser(args.model_checkpoint_path)
    if args.output_video_path:
        args.output_video_path = os.path.expanduser(args.output_video_path)
        os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)

    # Load model
    print("Loading DomainSeg model...")
    model = DomainSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("DomainSeg model loaded")

    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera_id}")

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Optional video writer
    writer = None
    if args.output_video_path:
        writer = cv2.VideoWriter(
            args.output_video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps_input,
            FRAME_ORI_SIZE
        )
        print(f"Recording output to: {args.output_video_path}")

    # Optional ffplay preview
    ffplay = None
    prev_time = 0.0
    if args.show:
        ffplay = subprocess.Popen(
            ["ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
             "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}", "-i", "pipe:0", "-loglevel", "quiet"],
            stdin=subprocess.PIPE
        )

    print("Starting camera inference (press Ctrl+C to quit)...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break

            # Resize for model inference
            image_inf = cv2.resize(frame, FRAME_INF_SIZE)
            image_pil = Image.fromarray(cv2.cvtColor(image_inf, cv2.COLOR_BGR2RGB))

            # Model inference
            with torch.no_grad():
                prediction = model.inference(image_pil)

            # Visualization
            vis = make_visualization(prediction)
            vis_resized = cv2.resize(vis, (frame.shape[1], frame.shape[0]))

            # Blend original frame and prediction
            blended = cv2.addWeighted(vis_resized, ALPHA_BLEND, frame, 1 - ALPHA_BLEND, 0)

            # Resize to output video size
            blended_out = cv2.resize(blended, FRAME_ORI_SIZE)

            # Write to video
            if writer:
                writer.write(blended_out)

            # Preview via ffplay
            if ffplay:
                now = time.time()
                if now - prev_time >= 1.0 / args.max_fps:
                    preview_frame = cv2.resize(blended_out, PREVIEW_SIZE)
                    ffplay.stdin.write(preview_frame.tobytes())
                    prev_time = now

    except KeyboardInterrupt:
        print("Camera inference stopped by user")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if ffplay:
        ffplay.stdin.close()
        ffplay.wait()

    print("Exited cleanly.")

if __name__ == "__main__":
    main()
# %%

