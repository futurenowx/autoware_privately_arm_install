#%%
import cv2
import sys
import numpy as np
from PIL import Image
import os
import cmapy
from argparse import ArgumentParser
import subprocess
import torch
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_3d_infer import Scene3DNetworkInfer

# Constants
OUTPUT_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 320)
INF_SIZE = (640, 320)
ALPHA_BLEND = 0.97  # transparency factor

def open_camera(camera_source):
    """
    Accepts:
      - integer camera id (0,1,2…)
      - /dev/videoX
      - RTSP/HTTP streams
      - video file
    """
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    print(f"Opening camera {camera_source}...")
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        raise RuntimeError(f"Error: Cannot open camera {camera_source}")

    return cap

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("--camera_id", default="/dev/video0",
                        help="Camera ID (0,1,2...) or device path (/dev/video0) or stream URL")
    parser.add_argument("-o", "--output_file", default=None)
    parser.add_argument("--skip_frames", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="Show live preview via ffplay")
    parser.add_argument("--max_fps", type=float, default=30.0, help="Preview FPS limit")
    args = parser.parse_args()

    # Load model
    print("Loading Scene3D model...")
    model = Scene3DNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("Scene3D Model Loaded")

    # Open camera
    try:
        cap = open_camera(args.camera_id)
    except Exception as e:
        print(e)
        return

    # Warm up camera
    print("Warming up camera...")
    for _ in range(20):
        cap.read()

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input is None or fps_input <= 0:
        fps_input = 30.0

    skip = args.skip_frames + 1
    fps_output = fps_input / skip

    # Optional video writer
    writer_obj = None
    if args.output_file:
        writer_obj = cv2.VideoWriter(
            args.output_file + ".avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps_output,
            OUTPUT_SIZE
        )
        print(f"Recording to {args.output_file}.avi")

    # ffplay preview
    ffplay = None
    if args.show:
        ffplay = subprocess.Popen(
            ["ffplay",
             "-f", "rawvideo",
             "-pixel_format", "bgr24",
             "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}",
             "-framerate", str(int(args.max_fps)),
             "-i", "pipe:0",
             "-loglevel", "quiet",
             "-autoexit"],
            stdin=subprocess.PIPE
        )

    prev_time = 0.0
    total_frames = 0
    frame_count = 0

    print("Processing started... Press Ctrl+C to quit.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None or frame.size == 0:
                print("Warning: Camera frame not ready or corrupted")
                time.sleep(0.01)
                continue

            total_frames += 1
            if total_frames % skip != 1:
                pass  # skip heavy processing but could still preview

            # Model inference
            try:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb).resize(INF_SIZE)

                prediction = model.inference(image_pil)
                prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
            except Exception as e:
                print(f"Model inference failed: {e}")
                continue

            # Normalize depth
            pred_img = 255.0 * (prediction - np.min(prediction)) / max((np.max(prediction) - np.min(prediction)), 1e-5)
            pred_img = pred_img.astype(np.uint8)

            # Apply colormap
            vis_obj = cv2.applyColorMap(pred_img, cmapy.cmap('viridis'))

            # Resize for output
            frame_hd = cv2.resize(frame, OUTPUT_SIZE)
            vis_obj_hd = cv2.resize(vis_obj, OUTPUT_SIZE)
            blended = cv2.addWeighted(vis_obj_hd, ALPHA_BLEND, frame_hd, 1 - ALPHA_BLEND, 0)

            # Write output
            if writer_obj:
                writer_obj.write(blended)

            # ffplay preview
            if args.show and ffplay:
                now = time.time()
                if now - prev_time >= 1.0 / args.max_fps:
                    preview_frame = cv2.resize(blended, PREVIEW_SIZE)
                    try:
                        ffplay.stdin.write(preview_frame.tobytes())
                    except BrokenPipeError:
                        ffplay = None
                    prev_time = now

            # FPS monitoring
            frame_count += 1
            t = time.time()
            if t - prev_time >= 1.0:
                print(f"FPS: {frame_count}")
                frame_count = 0
                prev_time = t

    except KeyboardInterrupt:
        print("Stopped by user.")

    # Cleanup
    cap.release()
    if writer_obj:
        writer_obj.release()
    if ffplay:
        ffplay.stdin.close()
        ffplay.wait()

    print("Processing completed. Preview closed automatically.")


if __name__ == "__main__":
    main()
#%%

