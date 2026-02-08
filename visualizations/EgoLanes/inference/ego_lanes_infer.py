import torch
from torchvision import transforms
import sys
sys.path.append("..")
from model_components.ego_lanes_network import EgoLanesNetwork


class EgoLanesNetworkInfer:
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
        self.model = EgoLanesNetwork()
        if checkpoint_path:
            print(f"Loading trained EgoLanes checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
        else:
            print("Loading vanilla EgoLanes model for training")

        # Move model to device and set eval mode
        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # Inference method
    # -------------------------
    def inference(self, image):
        # Convert image to tensor
        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)

        # Convert input to FP16 if model is FP16
        if next(self.model.parameters()).dtype == torch.half:
            image_tensor = image_tensor.half()

        # Run model
        prediction = self.model(image_tensor)

        # Convert output to numpy
        binary_seg = prediction.squeeze(0).cpu().detach().numpy()

        return binary_seg

