import os
import yaml
import torch
from ultralytics import YOLO
import kagglehub
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
AI_PROJECT_PATH = "/mnt/ai_data/aircraft_training_runs"
DATA_DIR = BASE_DIR / "data"
os.makedirs(AI_PROJECT_PATH, exist_ok=True)

os.environ['KAGGLEHUB_CACHE'] = str(DATA_DIR / "kaggle_cache")

AIRCRAFT_CLASSES = [
'A10', 'A400M', 'AG600', 'AH64', 'AKINCI', 'AV8B', 'An124', 'An22', 'An225', 'An72',
'B1', 'B2', 'B52', 'Be200', 'C1', 'C130', 'C17', 'C2', 'C390', 'C5', 'CH47', 'CH53',
'CL415', 'E2', 'E7', 'EF2000', 'EMB314', 'F117', 'F14', 'F15', 'F16', 'F18', 'F2',
'F22', 'F35', 'F4', 'FCK1', 'H6', 'Il76', 'J10', 'J20', 'J35', 'J36', 'JAS39',
'JF17', 'JH7', 'KAAN', 'KC135', 'KF21', 'KJ600', 'Ka27', 'Ka52', 'MQ9', 'Mi24',
'Mi26', 'Mi28', 'Mi8', 'Mig29', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale',
'SR71', 'Su24', 'Su25', 'Su34', 'Su47', 'Su57', 'TB001', 'TB2', 'Tejas', 'Tornado',
'Tu160', 'Tu22M', 'Tu95', 'U2', 'UH60', 'US2', 'V22', 'Vulcan', 'WZ7', 'X32',
'XB70', 'Y20', 'YF23', 'Z10', 'Z19'
]
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
        print(f"Dataset found at: {base_path}")
    else:
        print(f"Downloading from Kaggle...")
        path = kagglehub.dataset_download("ahnuf05/aeroscan-military-aircraft-classification")
        base_path = os.path.abspath(path)

    data_config = {
        'path': base_path,
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(AIRCRAFT_CLASSES),
        'names': AIRCRAFT_CLASSES
    }

    yaml_path = DATA_DIR / 'aircraft_config.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"Configuration saved to: {yaml_path}")

    model = YOLO('yolo26m.pt')

    print(f"Starting training on: {device_name} with imgsz=640")

    model.train(
        data=str(yaml_path),
        epochs=115,
        imgsz=640,
        batch=20,
        workers=12,
        device=target_device,
        amp=True,

        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5.0,
        label_smoothing=0.1,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        patience=25,
        project=AI_PROJECT_PATH,
        name='yolo26m_imgsz640_fixed_classes',
        exist_ok=True
    )

    print(f"Success! Results are in: {AI_PROJECT_PATH}")


if __name__ == '__main__':
    main()