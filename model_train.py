from ultralytics import YOLO

def main():
    MODEL_NAME = "yolov8m.pt"
    DATA_YAML = r"C:\project\vision\datasets\recycle\data\YoloSplit\data.yaml"

    model = YOLO(r"C:\Users\kbg00\runs\train_auto\exp\weights\epoch44.pt")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,          
        batch=24,           # 4080 VRAM 16GB 기준 v8m 모델 최적값
        workers=16,         # 7950X 최적값
        device=0,
        amp=True,           # FP16 자동
        project="runs/train_auto",
        name="exp",
        exist_ok=True,

        resume=True,
        save=True,
        save_period=1
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




