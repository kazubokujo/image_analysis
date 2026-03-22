from PIL import Image
import torch
import numpy as np


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Classifier:
    def __init__(self, model, processor, db):
        self.model = model
        self.processor = processor
        self.db = db

        
        self.labels = [
            "a photo of a person",
            "a close-up photo of a face",

            "a photo of a dog",
            "a photo of a cat",

            "a photo of food",
            "a meal on a table",

            "a landscape photo",
            "a photo of nature",

            "a photo of a building",
            "an urban scene",

            "a photo of plants",
            "a photo of flowers"
        ]
                   
    def clean_label(self, label):
        label = label.replace("a photo of ", "")
        label = label.replace("a ", "")
        label = label.replace("an ", "")
        label = label.replace("close-up ", "")
        label = label.replace("photo of ", "")
        label = label.replace("on a table", "food")

        return label.strip()
        
        
    def _clip_classify(self, image_path):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=self.labels,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1)

        return self.clean_label(self.labels[probs.argmax()])

    def _similarity_classify(self, feature):
        db_features, db_labels = self.db.get_labeled_features()

        best_score = 0
        best_label = "unknown"

        for f, label in zip(db_features, db_labels):
            score = cosine_similarity(feature, f)

            if score > best_score:
                best_score = score
                best_label = label

        if best_score < 0.3:
            return "unknown"

        return best_label

    def classify(self, image_path, feature):
        return self._clip_classify(image_path)