from ultralytics import YOLO
import cv2
import os

model_path = './weights/best.pt'
images_dir = './dataset/images/test/MOT20-04'
output_dir = './dataset/output/MOT20-04'

model = YOLO(model_path)

images = sorted([images_dir + '/' + image for image in os.listdir(images_dir) if image.endswith('.jpg')])

for image in images:
    model.track(image, tracker="bytetrack.yaml", save=True)
