import os
import torch
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / 'runs' / 'yolo26n_training' / 'weights' / 'best.pt'


def run_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please train the model first or provide the correct path to best.pt")
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
    print(f"Loading model from: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    source_path = BASE_DIR / 'test_images'

    if not os.path.exists(source_path):
        os.makedirs(source_path, exist_ok=True)
        print(f"NOTICE: Created '{source_path}' directory. Drop your images there and run the script again!")
        return

    if not any(os.scandir(source_path)):
        print(f"WARNING: The directory '{source_path}' is empty. Add some images to test the model.")
        return

    print(f"Running inference on images from: {source_path}")

    results = model.predict(
        source=str(source_path),
        save=True,
        conf=0.25,
        imgsz=640,
        device=target_device
    )

    print(f"\nInference finished successfully.")
    print(f"Check the results in the 'runs/detect/' directory.")


if __name__ == '__main__':
    run_inference()