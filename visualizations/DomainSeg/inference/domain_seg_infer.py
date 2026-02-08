#%%
import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from model_components.domain_seg_network import DomainSegNetwork

class DomainSegNetworkInfer:
    def __init__(self, checkpoint_path: str = ""):
        # -------------------------
        # Image preprocessing
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
        # Load model
        # -------------------------
        scene_net = SceneSegNetwork()
        self.model = DomainSegNetwork(scene_net)

        if checkpoint_path:
            print(f"Loading trained DomainSeg checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
        else:
            raise ValueError("No checkpoint path provided for DomainSeg model")

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # Inference
    # -------------------------
    def inference(self, image: Image.Image):
        # Check image size
        if image.size != (640, 320):
            image = image.resize((640, 320))

        # Convert image to tensor
        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)

        # Convert input to FP16 if model is half-precision
        if next(self.model.parameters()).dtype == torch.half:
            image_tensor = image_tensor.half()

        # Run model
        with torch.no_grad():
            prediction = self.model(image_tensor)

        # Process output to binary mask
        prediction = prediction.squeeze(0).cpu().detach()  # CxHxW
        prediction = prediction.permute(1, 2, 0)           # HxWxC
        output = prediction.numpy()
        output[output <= 0] = 0.0
        output[output > 0] = 1.0

        return output

