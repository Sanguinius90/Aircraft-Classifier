import os
import torch
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AI_DATA_RUNS = "/mnt/ai_data/aircraft_training_runs"
LAST_WEIGHTS = "/mnt/ai_data/aircraft_training_runs/yolo26m_imgsz640_fixed_classes/weights/last.pt"
DATA_CONFIG = str(BASE_DIR / 'data' / 'aircraft_config.yaml')

def resume_training():
    if not os.path.exists(LAST_WEIGHTS):
        print(f"ERROR: Checkpoint file not found at: {LAST_WEIGHTS}")
        return

    target_device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Resuming training from checkpoint: {LAST_WEIGHTS}")

    try:
        model = YOLO(LAST_WEIGHTS)

        model.train(resume=True)


    except Exception as e:
        print(f"An error occurred during training resumption: {e}")

if __name__ == '__main__':
    resume_training()