import numpy as np


def normalize(v):
    return v.flatten()


def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cluster_people(features, threshold=0.8, max_compare=50):
    clusters = []
    labels = []

    for f in features:
        f = normalize(f)

        assigned = False

        for i, cluster in enumerate(clusters[:max_compare]):
            sim = cosine_similarity(f, cluster)

            if sim > threshold:
                labels.append(i)
                assigned = True
                break

        if not assigned:
            clusters.append(f)
            labels.append(len(clusters) - 1)

    return labels