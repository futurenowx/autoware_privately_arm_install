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
from tqdm import tqdm  # <-- progress bar

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_seg_infer import SceneSegNetworkInfer

# -------------------------
OUTPUT_SIZE = (640, 400)
PREVIEW_SIZE = (640, 320)
ALPHA_BLEND = 0.5
# -------------------------

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

    model = SceneSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)

    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {args.video_filepath}")

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
    skip = args.skip_frames + 1
    fps_output = fps_input / skip

    output_obj_path = args.output_file + ".avi"
    output_free_path = args.output_file + "_freespace.avi"

    writer_obj = cv2.VideoWriter(output_obj_path, cv2.VideoWriter_fourcc(*"MJPG"), fps_output, OUTPUT_SIZE)
    writer_free = cv2.VideoWriter(output_free_path, cv2.VideoWriter_fourcc(*"MJPG"), fps_output, OUTPUT_SIZE)

    ffplay = None
    prev_time = 0.0
    if args.show:
        ffplay = subprocess.Popen(
            [
                "ffplay",
                "-loglevel", "quiet",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}",
                "-framerate", str(int(args.max_fps)),
                "-autoexit",  # automatically closes preview when stream ends
                "-"
            ],
            stdin=subprocess.PIPE
        )

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = total_frames_in_video // skip

    total_frames = 0
    with tqdm(total=frames_to_process, desc="Processing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            if total_frames % skip != 1:
                continue  # skip frames

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb).resize((640, 320))

            with torch.no_grad():
                prediction = model.inference(pil_image)

            vis_obj = make_visualization(prediction)
            vis_free = make_visualization_freespace(prediction, pil_image)

            frame_resized = cv2.resize(frame, OUTPUT_SIZE)
            vis_obj_resized = cv2.resize(vis_obj, OUTPUT_SIZE)
            vis_free_resized = cv2.resize(vis_free, OUTPUT_SIZE)

            blended_obj = cv2.addWeighted(vis_obj_resized, ALPHA_BLEND, frame_resized, 1 - ALPHA_BLEND, 0)
            blended_free = cv2.addWeighted(vis_free_resized, ALPHA_BLEND, frame_resized, 1 - ALPHA_BLEND, 0)

            writer_obj.write(blended_obj)
            writer_free.write(blended_free)

            if ffplay:
                now = time.time()
                if now - prev_time >= 1.0 / args.max_fps:
                    preview_frame = cv2.resize(blended_obj, PREVIEW_SIZE)
                    ffplay.stdin.write(preview_frame.tobytes())
                    prev_time = now

            pbar.update(1)

    # Release resources
    cap.release()
    writer_obj.release()
    writer_free.release()

    if ffplay:
        ffplay.stdin.close()  # signal end of frames
        ffplay.wait()         # wait until ffplay exits

    print("Processing completed. Preview closed automatically.")

if __name__ == "__main__":
    main()

