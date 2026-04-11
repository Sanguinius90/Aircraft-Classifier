import os
import yaml
import torch
from ultralytics import YOLO
import kagglehub
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

os.environ['KAGGLEHUB_CACHE'] = str(DATA_DIR / "kaggle_cache")


def main():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        target_device = 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_name = "Apple Silicon (MPS)"
        target_device = "mps"
    else:
        device_name = "CPU"
        target_device = "cpu"

    print(f"Detected device: {device_name}")

    WORKSTATION_PATH = "/mnt/ai_data/kaggle_cache/datasets/ahnuf05/aeroscan-military-aircraft-classification/versions/1"

    if os.path.exists(WORKSTATION_PATH):
        base_path = WORKSTATION_PATH
        print(f"Dataset found at workstation path: {base_path}")
    else:
        print(f"Workstation path not found. Checking Kaggle cache...")
        path = kagglehub.dataset_download("ahnuf05/aeroscan-military-aircraft-classification")
        base_path = os.path.abspath(path)
        print(f"Dataset located at: {base_path}")

    classes_file = os.path.join(base_path, 'classes.txt')
    class_names = []
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

    if not class_names:
        print("Classes file not found, generating generic names...")
        class_names = [f"Class_{i}" for i in range(150)]

    data_config = {
        'path': base_path,
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = DATA_DIR / 'aircraft_config.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    print(f"Configuration saved to: {yaml_path}")

    model = YOLO('yolo26n.pt')
    print(f"Starting training on: {device_name}")

    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=32,
        workers=12,
        device=target_device,
        amp=False,

        optimizer='AdamW',
        lr0=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        label_smoothing=0.1,
        augment=True,

        mosaic=1.0,
        mixup=0.2,
        patience=15,
        project=str(BASE_DIR / 'runs'),
        name='yolo26n_training',
        exist_ok=True
    )

    print(f"Success! Results are in: {BASE_DIR}/runs")


if __name__ == '__main__':
    main()