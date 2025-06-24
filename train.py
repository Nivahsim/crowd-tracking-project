# train_mot20_yolo11.py

from ultralytics import YOLO

yaml_path = './dataset/mot20.yaml'

model = YOLO('yolo11l.pt')

# Запуск обучения
model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    workers=8,
    optimizer='AdamW',
    device=0
)


