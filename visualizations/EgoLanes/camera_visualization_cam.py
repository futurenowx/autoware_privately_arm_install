import sys
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torch
import subprocess
import time

sys.path.append("../..")
from inference.ego_lanes_infer import EgoLanesNetworkInfer
from image_visualization import make_visualization

# -------------------------
# Frame sizes
# -------------------------
FRAME_INF_SIZE = (640, 320)  # for model inference
FRAME_ORI_SIZE = (720, 360)  # output video size
PREVIEW_SIZE = (720, 360)    # preview window size (match output for ffplay)


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True, help="Path to EgoLanes PyTorch checkpoint")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device index")
    parser.add_argument("-o", "--output_video_path", default=None, help="Optional output video path")
    parser.add_argument("--show", action="store_true", help="Show live preview using ffplay")
    parser.add_argument("--max-fps", type=float, default=30.0, help="Maximum preview FPS")
    args = parser.parse_args()

    # -------------------------
    # Load model (GPU + FP16)
    # -------------------------
    print("Loading EgoLanes model...")
    model = EgoLanesNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("EgoLanes model successfully loaded!")

    if "cuda" in str(model.device):
        model.model.half()  # FP16
        print("FP16 enabled for GPU inference")

    # -------------------------
    # Open camera
    # -------------------------
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback

    # -------------------------
    # Optional video writer
    # -------------------------
    writer = None
    if args.output_video_path:
        writer = cv2.VideoWriter(
            args.output_video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            FRAME_ORI_SIZE
        )
        print(f"Recording output to: {args.output_video_path}")

    # -------------------------
    # Launch ffplay for preview if requested
    # -------------------------
    if args.show:
        ffplay = subprocess.Popen(
            ["ffplay", "-f", "rawvideo", "-pixel_format", "bgr24",
             "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}", "-i", "pipe:0", "-loglevel", "quiet"],
            stdin=subprocess.PIPE
        )

    prev_time = 0.0
    print("Starting camera inference (press Ctrl+C to quit)")

    # -------------------------
    # Camera loop
    # -------------------------
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # Resize for model inference
            image = cv2.resize(frame, FRAME_INF_SIZE)

            # -------------------------
            # Model inference
            # -------------------------
            with torch.no_grad():
                prediction = model.inference(image)

            # -------------------------
            # Visualization
            # -------------------------
            vis_image = make_visualization(image.copy(), prediction)
            if not isinstance(vis_image, np.ndarray):
                vis_image = np.array(vis_image)

            # Resize to output resolution
            vis_image = cv2.resize(vis_image, FRAME_ORI_SIZE)

            # Write video if enabled
            if writer:
                writer.write(vis_image)

            # -------------------------
            # ffplay preview
            # -------------------------
            if args.show:
                now = time.time()
                if now - prev_time >= 1.0 / args.max_fps:
                    preview_frame = cv2.resize(vis_image, PREVIEW_SIZE)
                    ffplay.stdin.write(preview_frame.tobytes())
                    prev_time = now

    except KeyboardInterrupt:
        print("Camera preview stopped by user.")

    # -------------------------
    # Cleanup
    # -------------------------
    cap.release()
    if writer:
        writer.release()
    if args.show:
        ffplay.stdin.close()
        ffplay.wait()

    print("Exited cleanly.")


if __name__ == "__main__":
    main()

