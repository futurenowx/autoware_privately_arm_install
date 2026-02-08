# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3

import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

sys.path.append('..')
from Models.model_components.ego_lanes_network import EgoLanesNetwork
from Models.model_components.auto_steer_network import AutoSteerNetwork


class AutoSpeedNetworkInfer:
    def __init__(self, egolanes_checkpoint_path='', autosteer_checkpoint_path=''):

        # -------------------------
        # Image loader
        # -------------------------
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        # -------------------------
        # Device
        # -------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for inference")

        # -------------------------
        # Load models
        # -------------------------
        if not egolanes_checkpoint_path or not autosteer_checkpoint_path:
            raise ValueError("Checkpoint paths must be provided")

        # Ego lanes network
        self.egoLanesNetwork = EgoLanesNetwork()
        checkpoint = torch.load(egolanes_checkpoint_path, map_location=self.device)
        # Only load weights (state dict)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.egoLanesNetwork.load_state_dict(checkpoint)
        self.egoLanesNetwork.to(self.device).eval()
        

        # Auto steer network
        
        self.model = AutoSteerNetwork()
        checkpoint = torch.load(autosteer_checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device).eval()
        

        # -------------------------
        # Feature buffer
        # -------------------------
        self.feature = torch.zeros(1, 64, 10, 20, device=self.device)

        # -------------------------
        # Previous frame buffer
        # -------------------------
        self.image_T_minus_1 = Image.new("RGB", (640, 320), color=(0, 0, 0))

    # -------------------------
    # Inference
    # -------------------------
    def inference(self, image):

        width, height = image.size
        if width != 640 or height != 320:
            raise ValueError("Input image must be 640x320 (WxH)")

        # Load images
        image_tensor_T_minus_1 = (
            self.image_loader(self.image_T_minus_1)
            .unsqueeze(0)
            .to(self.device)
        )

        image_tensor_T = (
            self.image_loader(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # FP16 safety (match model precision)
        if next(self.egoLanesNetwork.parameters()).dtype == torch.half:
            image_tensor_T_minus_1 = image_tensor_T_minus_1.half()
            image_tensor_T = image_tensor_T.half()

        # -------------------------
        # Run inference
        # -------------------------
        with torch.no_grad():
            l1 = self.egoLanesNetwork(image_tensor_T_minus_1)
            l2 = self.egoLanesNetwork(image_tensor_T)

            lane_features_concat = torch.cat((l1, l2), dim=1)

            _, prediction = self.model(lane_features_concat)

            prediction = prediction.squeeze(0).cpu()

        # Convert class index to steering angle
        output = torch.argmax(prediction).item() - 30

        # Update previous frame
        self.image_T_minus_1 = image.copy()

        return output

