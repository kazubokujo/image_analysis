import face_recognition
import numpy as np


def extract_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        return encodings[0]

    return None


def face_distance(a, b):
    return np.linalg.norm(a - b)