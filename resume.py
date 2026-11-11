import os
import torch
from ultralytics import YOLO
from pathlib import Path

from augmentations import build_train_augmentations

BASE_DIR = Path(__file__).resolve().parent
AI_DATA_RUNS = "/mnt/ai_data/aircraft_training_runs"
LAST_WEIGHTS = "/mnt/ai_data/aircraft_training_runs/yolo26m_imgsz640_augmented/weights/last.pt"
DATA_CONFIG = str(BASE_DIR / 'data' / 'aircraft_config.yaml')

def resume_training():
    if not os.path.exists(LAST_WEIGHTS):
        print(f"ERROR: Checkpoint file not found at: {LAST_WEIGHTS}")
        return

    target_device = 0 if torch.cuda.is_available() else "cpu"
    train_augmentations = build_train_augmentations()
    print(f"Resuming training from checkpoint: {LAST_WEIGHTS}")
    print(
        "Using Albumentations transforms: "
        + ", ".join(type(transform).__name__ for transform in train_augmentations)
    )

    try:
        model = YOLO(LAST_WEIGHTS)

        model.train(
            resume=LAST_WEIGHTS,
            augment=True,
            augmentations=train_augmentations,
            device=target_device,
        )


    except Exception as e:
        print(f"An error occurred during training resumption: {e}")

if __name__ == '__main__':
    resume_training()