from ultralytics import YOLO

def main():
    MODEL_NAME = "yolov8n.pt"
    DATA_YAML = r"C:\project\vision\datasets\recycle\data\YoloSplit\data.yaml"

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=512,          # 640보다 40% 빠름
        batch=32,           # 4080 VRAM 16GB 기준 안정적   
        workers=24,         # 7950X 최적값
        device=0,
        amp=True,           # FP16 자동
        project="runs/train_auto",
        name="exp",
        exist_ok=True,
        cache='disk'
    )

    model = YOLO("runs/train_auto/exp/weights/best.pt")
    metrics = model.val(device=0)

    print("=== YOLO 성능 지표 ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()




