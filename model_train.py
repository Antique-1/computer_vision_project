import os
import yaml
from glob import glob
from ultralytics import YOLO
import shutil
from sklearn.model_selection import train_test_split

# 설정
DATASET_DIR = r"C:\project\vision\datasets\recycle\data\Training"
MODEL_NAME = "yolov8n.pt"
OUTPUT_DIR = "runs/train_auto"


def get_classes(label_dir):
    """YOLO 라벨 txt 파일들을 읽어 클래스 ID 목록 자동 추출 (하위 폴더 포함)"""
    class_ids = set()
    label_files = glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)

    for file in label_files:
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    if len(parts) > 0:
                        class_ids.add(int(parts[0]))

    class_list = sorted(list(class_ids))
    return [f"class_{i}" for i in class_list]


def create_dataset_split(img_dir, label_dir):
    """라벨(txt) 있는 이미지만 train/val로 분리"""

    # -----------------------------
    # 🔥 라벨이 있는 이미지 목록만 추출
    # -----------------------------
    all_images = glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)

    valid_images = []
    for img in all_images:
        txt = img.replace("\\images\\", "\\labels_yolo\\").replace(".jpg", ".txt")
        txt = txt.replace("/images/", "/labels_yolo/")
        if os.path.exists(txt):   # 라벨이 있는 경우만 포함
            valid_images.append(img)

    print(f"[INFO] 전체 이미지: {len(all_images)} → 라벨 있는 이미지만 사용: {len(valid_images)}")

    if len(valid_images) == 0:
        raise ValueError("⚠ 라벨(txt) 있는 이미지가 없습니다. 변환 과정을 다시 확인하세요.")

    # train/val split
    train_imgs, val_imgs = train_test_split(valid_images, test_size=0.2, random_state=42)

    # train/val 폴더 생성
    split_dir = os.path.join(DATASET_DIR, "split")
    train_img_dir = os.path.join(split_dir, "images/train")
    val_img_dir = os.path.join(split_dir, "images/val")
    train_lbl_dir = os.path.join(split_dir, "labels/train")
    val_lbl_dir = os.path.join(split_dir, "labels/val")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    def copy_image_and_label(img_path, dest_img_dir, dest_lbl_dir):
        shutil.copy(img_path, dest_img_dir)

        txt = img_path.replace("\\images\\", "\\labels_yolo\\").replace(".jpg", ".txt")
        if os.path.exists(txt):
            shutil.copy(txt, dest_lbl_dir)

    # copy train
    for img in train_imgs:
        copy_image_and_label(img, train_img_dir, train_lbl_dir)

    # copy val
    for img in val_imgs:
        copy_image_and_label(img, val_img_dir, val_lbl_dir)

    return train_img_dir, val_img_dir


def create_yaml(train_path, val_path, classes, yaml_path="dataset.yaml"):
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
    label_dir = os.path.join(DATASET_DIR, "labels_yolo")  

    assert os.path.exists(img_dir), f"{img_dir} 폴더가 없습니다!"
    assert os.path.exists(label_dir), f"{label_dir} 폴더가 없습니다!"

    # 클래스 자동 추출
    print("[INFO] 라벨 파일에서 클래스 자동 추출 중...")
    class_names = get_classes(label_dir)
    print(f"[INFO] 감지된 클래스: {class_names}")

    # train/val 분리 생성 (라벨 없는 이미지는 자동 제외)
    train_path, val_path = create_dataset_split(img_dir, label_dir)

    # yaml 생성
    yaml_path = "dataset.yaml"
    create_yaml(train_path, val_path, class_names, yaml_path)

    # YOLO 학습 시작
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




