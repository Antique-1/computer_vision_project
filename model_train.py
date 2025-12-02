from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
DATA_YAML = r"C:\project\vision\datasets\recycle\data\YoloSplit\data.yaml"

model = YOLO(MODEL_NAME)

model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/train_auto",
    name="exp",
    exist_ok=True
)






