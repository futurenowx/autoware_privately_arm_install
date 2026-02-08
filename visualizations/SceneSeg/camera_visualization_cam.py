#%%
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import subprocess
import torch
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_seg_infer import SceneSegNetworkInfer

OUTPUT_SIZE = (640, 400)
FRAME_INF_SIZE = (640, 320)
PREVIEW_SIZE = (640, 320)
ALPHA_BLEND = 0.5


def find_freespace_edge(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        return max(contours, key=lambda x: cv2.contourArea(x))
    return None


def make_visualization_freespace(prediction, image):
    colour_mask = np.array(image)
    free_space_labels = np.where(prediction == 2)
    row, col = prediction.shape
    binary_mask = np.zeros((row, col), dtype="uint8")
    binary_mask[free_space_labels[0], free_space_labels[1]] = 255
    edge_contour = find_freespace_edge(binary_mask)
    if edge_contour is not None and edge_contour.size > 0:
        cv2.fillPoly(colour_mask, pts=[edge_contour], color=(28, 255, 145))
    colour_mask = cv2.cvtColor(colour_mask, cv2.COLOR_RGB2BGR)
    return colour_mask


def make_visualization(prediction):
    row, col = prediction.shape
    vis = np.zeros((row, col, 3), dtype="uint8")
    vis[:, :, 0] = 255
    vis[:, :, 1] = 93
    vis[:, :, 2] = 61
    fg = np.where(prediction == 1)
    vis[fg[0], fg[1], 0] = 145
    vis[fg[0], fg[1], 1] = 28
    vis[fg[0], fg[1], 2] = 255
    return vis


def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("-o", "--output_file", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max_fps", type=float, default=30.0)
    args = parser.parse_args()

    # -------------------------
    # Load model
    # -------------------------
    print("Loading model...")
    model = SceneSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)

    if "cuda" in str(model.device):
        model.model.half()
        print("FP16 enabled for GPU inference")

    # -------------------------
    # Open camera (same as working example)
    # -------------------------
    print("Opening camera...")
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print("Error opening camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    # -------------------------
    # Video writers
    # -------------------------
    writer_obj = None
    writer_free = None

    if args.output_file:
        writer_obj = cv2.VideoWriter(
            args.output_file + ".avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            OUTPUT_SIZE
        )

        writer_free = cv2.VideoWriter(
            args.output_file + "_freespace.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            OUTPUT_SIZE
        )

    # -------------------------
    # ffplay preview
    # -------------------------
    ffplay = None
    if args.show:
        ffplay = subprocess.Popen(
            ["ffplay",
             "-f", "rawvideo",
             "-pixel_format", "bgr24",
             "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}",
             "-i", "pipe:0",
             "-loglevel", "quiet"],
            stdin=subprocess.PIPE
        )

    prev_time = 0.0

    print("Starting camera inference (Ctrl+C to quit)")

    # -------------------------
    # Camera loop
    # -------------------------
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Do not exit immediately; cameras sometimes drop frames
                time.sleep(0.01)
                continue

            # Resize for inference
            image_resized = cv2.resize(frame, FRAME_INF_SIZE)

            # Convert to PIL for model
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Inference
            with torch.no_grad():
                prediction = model.inference(pil_image)

            # Visualization
            vis_obj = make_visualization(prediction)
            vis_free = make_visualization_freespace(prediction, pil_image)

            vis_obj = cv2.resize(vis_obj, OUTPUT_SIZE)
            vis_free = cv2.resize(vis_free, OUTPUT_SIZE)
            frame_out = cv2.resize(frame, OUTPUT_SIZE)

            blended_obj = cv2.addWeighted(vis_obj, ALPHA_BLEND, frame_out, 1 - ALPHA_BLEND, 0)
            blended_free = cv2.addWeighted(vis_free, ALPHA_BLEND, frame_out, 1 - ALPHA_BLEND, 0)

            if writer_obj:
                writer_obj.write(blended_obj)
                writer_free.write(blended_free)

            # Preview
            if args.show:
                now = time.time()
                if now - prev_time >= 1.0 / args.max_fps:
                    preview_frame = cv2.resize(blended_obj, PREVIEW_SIZE)
                    ffplay.stdin.write(preview_frame.tobytes())
                    prev_time = now

    except KeyboardInterrupt:
        print("Stopped by user.")

    # Cleanup
    cap.release()
    if writer_obj:
        writer_obj.release()
        writer_free.release()
    if ffplay:
        ffplay.stdin.close()
        ffplay.wait()

    print("Exited cleanly.")


if __name__ == "__main__":
    main()

