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

# 학습된 모델 불러오기
model = YOLO("runs/train_auto/exp/weights/best.pt")

# 검증 데이터로 성능 평가
metrics = model.val()

print("=== YOLO 성능 지표 ===")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")




