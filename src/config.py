import os

BASE_DIR = "/app"

INPUT_DIR = os.path.join(BASE_DIR, "data/input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/output")
WORK_DIR = os.path.join(BASE_DIR, "data/work")

DB_PATH = os.path.join(BASE_DIR, "db/images.db")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

SIMILARITY_THRESHOLD = 0.35
FACE_THRESHOLD = 0.5

BATCH_SIZE = 100