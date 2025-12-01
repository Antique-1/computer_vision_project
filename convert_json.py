import os
import json
import glob

# ---------------------------------
# 1. 클래스 이름 → class_id 매핑
# ---------------------------------
CLASS_MAP = {
    "금속캔": 0,
    "종이": 1,
    "페트병": 2,
    "플라스틱": 3,
    "스티로품": 4,
    "비닐": 5,
    "유리병": 6
}

# ---------------------------------
# 2. 이미지 파일 위치 자동 탐색 함수
# ---------------------------------
def find_image(images_root, filename):
    """
    images/ 모든 하위 폴더를 뒤져서 filename과 일치하는 이미지를 찾는다.
    """
    pattern = os.path.join(images_root, "**", filename)
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


# ---------------------------------
# 3. JSON → YOLO txt 변환 함수
# ---------------------------------
def convert_json_to_yolo(json_path, images_root, labels_root):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_info = data["IMAGE_INFO"]
    annos = data["ANNOTATION_INFO"]

    img_name = img_info["FILE_NAME"]
    img_width = img_info["IMAGE_WIDTH"]
    img_height = img_info["IMAGE_HEIGHT"]

    # 이미지 실제 경로 찾기
    image_path = find_image(images_root, img_name)
    if not image_path:
        print(f"[WARN] 이미지 없음 → {img_name}, JSON 경로: {json_path}")
        return

    # → 예: images/plastic01/파일1.jpg → plastic01
    class_folder = os.path.basename(os.path.dirname(image_path))

    # labels/plastic01 자동 생성
    label_dir = os.path.join(labels_root, class_folder)
    os.makedirs(label_dir, exist_ok=True)

    txt_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))

    lines = []

    for anno in annos:
        class_name = anno["CLASS"]

        if class_name not in CLASS_MAP:
            print(f"[WARN] CLASS_MAP에 없는 클래스: {class_name}")
            continue

        class_id = CLASS_MAP[class_name]

        points = anno["POINTS"][0]

        xc = points[0] / img_width
        yc = points[1] / img_height
        w = points[2] / img_width
        h = points[3] / img_height

        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # TXT 저장
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] {json_path} → {txt_path}")


# ---------------------------------
# 4. json 폴더 전체 변환 실행
# ---------------------------------
def convert_all_json(json_folder, images_root="images", labels_root="labels"):
    # 하위 폴더 포함 모든 JSON 파일 탐색
    json_files = glob.glob(os.path.join(json_folder, "**", "*.json"), recursive=True)

    if not json_files:
        print("❌ JSON 파일이 없습니다. 경로 확인하세요:", json_folder)
        return

    print(f"🔎 총 {len(json_files)}개의 JSON 라벨 발견")

    for json_path in json_files:
        convert_json_to_yolo(json_path, images_root, labels_root)

# 실행 예시
convert_all_json(
    r"C:\project\vision\datasets\recycle\data\Training\labels",
    images_root=r"C:\project\vision\datasets\recycle\data\Training\images",
    labels_root=r"C:\project\vision\datasets\recycle\data\Training\labels_yolo"
)
