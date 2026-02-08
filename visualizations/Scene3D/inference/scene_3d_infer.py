#%%
#! /usr/bin/env python3
import torch
from torchvision import transforms
import sys

sys.path.append("..")

from model_components.scene_seg_network import SceneSegNetwork
from model_components.scene_3d_network import Scene3DNetwork


class Scene3DNetworkInfer:
    def __init__(self, checkpoint_path: str = ""):

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
        # Device (GPU vs CPU)
        # -------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for inference")

        # -------------------------
        # Initialize model
        # -------------------------
        sceneSegNetwork = SceneSegNetwork()
        self.model = Scene3DNetwork(sceneSegNetwork)

        if checkpoint_path:
            print(f"Loading trained Scene3D checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
        else:
            raise ValueError("No checkpoint path provided")

        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # Inference method
    # -------------------------
    def inference(self, image):

        width, height = image.size
        if width != 640 or height != 320:
            raise ValueError("Input image must be 640x320 (WxH)")

        # Convert image to tensor
        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)

        # Convert input to FP16 if model is FP16
        if next(self.model.parameters()).dtype == torch.half:
            image_tensor = image_tensor.half()

        # Run model safely
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Convert output to numpy
        prediction = prediction.squeeze(0).cpu().detach()
        prediction = prediction.permute(1, 2, 0)  # HWC
        output_np = prediction.numpy()

        return output_np

