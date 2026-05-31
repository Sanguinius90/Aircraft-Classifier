import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

model = YOLO('/mnt/ai_data/aircraft_training_runs/yolo26m_imgsz640_fixed_classes/weights/best.pt')

img_path = 'test_images/0db5bf446bc545c2b4f1ab7baab8258d.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))
rgb_img = np.float32(img) / 255
input_tensor = torch.from_numpy(np.transpose(rgb_img, (2, 0, 1))).unsqueeze(0).to(torch.float32)
input_tensor.requires_grad = True

target_layers = [model.model.model[22]]

class YoloClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        preds = self.model(x)

        details = preds[1]

        if 'one2many' in details and 'scores' in details['one2many']:
            scores = details['one2many']['scores']
            class_scores = scores.max(dim=1)[0]
            return class_scores

        detections = preds[0]
        return detections[:, :, 4:].max(dim=1)[0]

model_wrapped = YoloClassifierWrapper(model.model)

cam = GradCAM(model=model_wrapped, target_layers=target_layers)

targets = [ClassifierOutputTarget(41)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

cv2.imwrite('gradcam_result.jpg', visualization)
print("Grad-CAM result saved as gradcam_result.jpg")