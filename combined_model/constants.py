import os

# Too much processes didn't show improvements
MAX_PROC = min(os.cpu_count(), 8)

MODEL_NAME = "google/muril-base-cased"

DATASET_FOLDER = "datasets/"
