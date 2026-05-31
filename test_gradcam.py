from ultralytics import YOLO
import torch

model = YOLO('/mnt/ai_data/aircraft_training_runs/yolo26m_imgsz640_fixed_classes/weights/best.pt')

results = model.predict(
    source='test_images/0db5bf446bc545c2b4f1ab7baab8258d.jpg',
    visualize=True,
    save=True,
    conf=0.25
)

print("Check the runs/detect/predict/visualize folder to see the features!")