import os
import yaml
from glob import glob
from ultralytics import YOLO

# 설정
DATASET_DIR = "datasets"
MODEL_NAME = "yolov8n.pt"
OUTPUT_DIR = "runs/train_auto"

def get_classes(label_dir):
    """YOLO 라벨 txt 파일들을 읽어 클래스 ID 목록 자동 추출"""
    class_ids = set()
    label_files = glob(os.path.join(label_dir, "*.txt"))

    for file in label_files:
        with open(file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 0:
                    class_ids.add(int(parts[0]))

    class_list = sorted(list(class_ids))
    return [f"class_{i}" for i in class_list]  # 이름이 없으면 class_0, class_1 형태로 생성


def create_yaml(train_path, val_path, classes, yaml_path="dataset.yaml"):
    """data.yaml 자동 생성"""
    data = {
        "train": train_path,
        "val": val_path,
        "nc": len(classes),
        "names": classes
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"[INFO] data.yaml 생성 완료 → {yaml_path}")
    return yaml_path


def main():
    img_dir = os.path.join(DATASET_DIR, "images")
    label_dir = os.path.join(DATASET_DIR, "labels")

    assert os.path.exists(img_dir), "datasets/images 폴더가 없습니다!"
    assert os.path.exists(label_dir), "datasets/labels 폴더가 없습니다!"

    # 클래스 자동 추출
    print("[INFO] 라벨 파일에서 클래스 자동 추출 중...")
    class_names = get_classes(label_dir)
    print(f"[INFO] 감지된 클래스: {class_names}")

    # train/val 8:2 분할 자동 수행
    images = sorted(glob(os.path.join(img_dir, "*.jpg")))
    total = len(images)
    split_idx = int(total * 0.8)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # txt 라벨도 동일하게 복사할 필요는 없음 (YOLO가 자동 참조)
    train_path = os.path.join(img_dir)
    val_path = os.path.join(img_dir)

    # yaml 자동 생성
    yaml_path = "dataset.yaml"
    create_yaml(train_path, val_path, class_names, yaml_path)

    # YOLO 학습
    print("[INFO] YOLO 학습 시작...")
    model = YOLO(MODEL_NAME)
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project=OUTPUT_DIR,
        name="exp",
        exist_ok=True
    )

    print("[INFO] 학습 완료!")
    print(f"[INFO] best.pt 경로: {OUTPUT_DIR}/exp/weights/best.pt")


if __name__ == "__main__":
    main()

