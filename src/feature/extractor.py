import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class FeatureExtractor:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def extract(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt")

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            # ✅ ここだけsqueeze
            return features[0].cpu().numpy()

        except Exception as e:
            print(f"Error: {image_path} -> {e}")
            return None