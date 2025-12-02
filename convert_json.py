import os
import json
import glob
from pathlib import Path

# 사용자 설정
JSON_FOLDER = r"C:\project\vision\datasets\recycle\data\Training\labels"
IMAGES_ROOT = r"C:\project\vision\datasets\recycle\data\Training\images"
LABELS_ROOT = r"C:\project\vision\datasets\recycle\data\Training\labels_yolo"  # 출력 (bbox)
CLASS_MAP = {
    "금속캔": 0,
    "종이": 1,
    "페트병": 2,
    "플라스틱": 3,
    "스티로폼": 4,
    "비닐": 5,
    "유리병": 6
}

# 이미지 찾기
def find_image(images_root, filename):
    pattern = os.path.join(images_root, "**", filename)
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

# POINTS -> flat numeric list
def extract_numeric_points(raw_pts):
    if raw_pts is None:
        return None
    if isinstance(raw_pts, list) and len(raw_pts) > 0 and isinstance(raw_pts[0], (list, tuple)):
        first = raw_pts[0]
        if all(isinstance(x, (int, float)) for x in first) and len(raw_pts) == 1:
            flat = [float(x) for x in first]
            return flat if len(flat) >= 4 else None
        flat = []
        for pair in raw_pts:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                try:
                    flat.append(float(pair[0])); flat.append(float(pair[1]))
                except Exception:
                    continue
        return flat if len(flat) >= 4 else None
    if isinstance(raw_pts, list) and len(raw_pts) >= 4 and all(isinstance(x, (int, float)) for x in raw_pts):
        return [float(x) for x in raw_pts]
    if isinstance(raw_pts, list) and len(raw_pts) >= 4:
        try:
            return [float(x) for x in raw_pts]
        except Exception:
            return None
    return None

def flat_pts_to_bbox(flat_pts):
    n = len(flat_pts)
    if n == 4:
        a,b,c,d = flat_pts[0], flat_pts[1], flat_pts[2], flat_pts[3]
        # heuristic: if c>a and d>b -> [xmin,ymin,xmax,ymax]
        if c > a and d > b:
            xmin, ymin, xmax, ymax = a,b,c,d
            w = xmax - xmin; h = ymax - ymin
            xc = xmin + w/2.0; yc = ymin + h/2.0
            return xc, yc, w, h
        else:
            # treat as x,y,w,h (top-left)
            xc = a + c/2.0 if c >= 0 else a
            yc = b + d/2.0 if d >= 0 else b
            return xc, yc, c, d
    elif n > 4:
        xs = flat_pts[0::2]; ys = flat_pts[1::2]
        xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
        w = xmax - xmin; h = ymax - ymin
        xc = xmin + w/2.0; yc = ymin + h/2.0
        return xc, yc, w, h
    return None

def convert_json_to_yolo_robust(json_path, images_root, labels_root, stats):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        stats['json_read_error'].append((json_path, str(e)))
        return

    img_info = data.get("IMAGE_INFO", {})
    annos = data.get("ANNOTATION_INFO", [])

    img_name = img_info.get("FILE_NAME")
    img_w = img_info.get("IMAGE_WIDTH")
    img_h = img_info.get("IMAGE_HEIGHT")

    if not img_name or not img_w or not img_h:
        stats['bad_image_info'].append(json_path); return

    image_path = find_image(images_root, img_name)
    if not image_path:
        stats['image_not_found'].append(json_path); return

    folder = os.path.basename(os.path.dirname(image_path))
    label_dir = os.path.join(labels_root, folder)
    os.makedirs(label_dir, exist_ok=True)
    txt_path = os.path.join(label_dir, Path(img_name).stem + ".txt")

    out_lines = []
    for anno in annos:
        class_name = anno.get("CLASS")
        if class_name not in CLASS_MAP:
            stats['unknown_class'].append((json_path, class_name)); continue
        class_id = CLASS_MAP[class_name]

        raw_pts = anno.get("POINTS")
        flat = extract_numeric_points(raw_pts)
        if flat is None:
            stats['invalid_points'].append((json_path, raw_pts)); continue
        if len(flat) == 2:
            stats['single_point'].append((json_path, flat)); continue

        bbox = flat_pts_to_bbox(flat)
        if bbox is None:
            stats['bbox_fail'].append((json_path, flat)); continue

        xc_abs, yc_abs, w_abs, h_abs = bbox
        if w_abs <= 0 or h_abs <= 0:
            xs = flat[0::2]; ys = flat[1::2]
            if len(xs) >= 2:
                xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
                w_abs = xmax - xmin; h_abs = ymax - ymin
                xc_abs = xmin + w_abs/2.0; yc_abs = ymin + h_abs/2.0

        if w_abs <= 0 or h_abs <= 0:
            stats['degenerate_bbox'].append((json_path, flat)); continue

        xc = max(0.0, min(1.0, float(xc_abs) / float(img_w)))
        yc = max(0.0, min(1.0, float(yc_abs) / float(img_h)))
        w = max(0.0, min(1.0, float(w_abs) / float(img_w)))
        h = max(0.0, min(1.0, float(h_abs) / float(img_h)))

        out_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        stats['converted'] += 1

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
        stats['files_written'] += 1
    except Exception as e:
        stats['write_error'].append((txt_path, str(e)))

def run_all(json_folder, images_root, labels_root):
    json_files = glob.glob(os.path.join(json_folder, "**", "*.json"), recursive=True)
    print(f"전체 JSON 파일 수: {len(json_files)}")

    stats = {
        'converted': 0,
        'files_written': 0,
        'json_read_error': [], 'bad_image_info': [],
        'image_not_found': [], 'unknown_class': [],
        'invalid_points': [], 'single_point': [],
        'bbox_fail': [], 'degenerate_bbox': [], 'write_error': []
    }

    for j in json_files:
        convert_json_to_yolo_robust(j, images_root, labels_root, stats)

    print("\n===== 변환 요약 =====")
    for k in ['converted','files_written','json_read_error','bad_image_info','image_not_found','unknown_class','invalid_points','single_point','bbox_fail','degenerate_bbox','write_error']:
        v = stats[k] if k in stats else None
        if isinstance(v, list):
            print(f"{k}: {len(v)}")
        else:
            print(f"{k}: {v}")

    problem_dir = os.path.join(labels_root, "_problems")
    os.makedirs(problem_dir, exist_ok=True)
    def save_list(name, data):
        p = os.path.join(problem_dir, name + ".txt")
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(str(item) + "\n")
        print(f"Saved {name}: {len(data)} -> {p}")

    for name in ['json_read_error','bad_image_info','image_not_found','unknown_class','invalid_points','single_point','bbox_fail','degenerate_bbox','write_error']:
        save_list(name, stats[name])

    return stats

if __name__ == "__main__":
    stats = run_all(JSON_FOLDER, IMAGES_ROOT, LABELS_ROOT)
    print("모든 JSON 변환 완료. 이제 prepare_yolo_dataset.py로 split 진행하세요.")
