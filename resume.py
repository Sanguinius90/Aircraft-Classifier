import os
import torch
from ultralytics import YOLO
from pathlib import Path

# --- DYNAMIC PATH CONFIGURATION ---
# Detect project root directory
BASE_DIR = Path(__file__).resolve().parent

LAST_WEIGHTS = BASE_DIR / 'runs' / 'yolo26n_training' / 'weights' / 'last.pt'


def resume_training():
    if not os.path.exists(LAST_WEIGHTS):
        print(f"ERROR: Checkpoint file not found at: {LAST_WEIGHTS}")
        print("Hint: Ensure that the initial training has started and created the 'runs' directory.")
        return

    if torch.cuda.is_available():
        target_device = 0
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        target_device = "mps"
        device_name = "Apple Silicon (MPS)"
    else:
        target_device = "cpu"
        device_name = "CPU"

    print(f"Detected device: {device_name}")
    print(f"Resuming training from checkpoint: {LAST_WEIGHTS}")

    try:
        model = YOLO(str(LAST_WEIGHTS))

        model.train(resume=True, device=target_device)

    except Exception as e:
        print(f"An error occurred during training resumption: {e}")


if __name__ == '__main__':
    resume_training()