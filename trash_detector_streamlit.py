import streamlit as st
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import json
import os

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# -----------------------------
# 기본 분리수거 가이드 매핑 (예시)
# -----------------------------
DEFAULT_GUIDE = {
    "plastic": "내용물 비우기 → 라벨 제거 후 분리배출",
    "glass": "뚜껑 분리 → 색상별 분리 배출",
    "can": "내용물 비우기 → 압착 후 배출",
    "paper": "이물질 제거 → 건조 후 배출",
    "food": "음식물 쓰레기(물기 제거 후 전용 봉투에)",
    "vinyl": "오염 여부 확인 → 일반(비닐) 분류",
}

# -----------------------------
# 유틸리티: 더미 예측(ultralytics가 없을 때)
# -----------------------------

def dummy_predict(img: Image.Image, conf_thres: float = 0.25) -> List[Dict[str, Any]]:
    """더미 예측: 테스트용으로 이미지 중앙에 하나의 'plastic' 바운딩박스 반환"""
    w, h = img.size
    cx, cy = w // 2, h // 2
    box_w, box_h = w // 3, h // 3
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w, cx + box_w // 2)
    y2 = min(h, cy + box_h // 2)
    return [{
        "class": "plastic",
        "score": 0.85,
        "bbox": [x1, y1, x2, y2]
    }]

# -----------------------------
# 모델 로드 및 예측 래퍼
# -----------------------------

MODEL_CACHE = {}

def load_yolo_model(model_path: str = "yolov8n.pt", device: str = "cpu"):
    """ultralytics YOLO 모델 로드(캐시 사용)"""
    if not HAS_ULTRALYTICS:
        st.warning("ultralytics 패키지를 찾을 수 없습니다. 더미 예측으로 동작합니다.")
        return None
    key = (model_path, device)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    try:
        model = YOLO(model_path)
        # ultralytics 라이브러리는 predict 시 device 인자를 받음, 여기서는 모델 객체만 반환
        MODEL_CACHE[key] = model
        return model
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None


def predict_image_with_model(model, img: Image.Image, device: str = "cpu", conf: float = 0.25) -> List[Dict[str, Any]]:
    """ultralytics 모델로 예측 수행하여 통일된 포맷으로 반환
    반환 포맷: [{class: str, score: float, bbox: [x1,y1,x2,y2]}, ...]
    """
    if model is None or not HAS_ULTRALYTICS:
        return dummy_predict(img, conf)

    # ultralytics 모델은 numpy array 입력을 받음
    arr = np.array(img.convert("RGB"))
    results = model.predict(source=arr, conf=conf, device=0 if device.startswith("cuda") else "cpu")
    out = []
    # results는 list 형태로 반환
    try:
        r = results[0]
        boxes = r.boxes
        cls_names = model.names if hasattr(model, 'names') else {}
        for box in boxes:
            xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = cls_names.get(cls_id, str(cls_id))
            out.append({"class": cls_name, "score": score, "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]})
    except Exception as e:
        st.error(f"예측 후 결과 파싱 실패: {e}")
    return out

# -----------------------------
# 시각화: 바운딩 박스 그리기
# -----------------------------

def draw_boxes(img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    img2 = img.convert("RGBA")
    draw = ImageDraw.Draw(img2)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        score = det["score"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
        text = f"{cls} {score:.2f}"
        text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=(255, 0, 0, 200))
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255, 255), font=font)

    return img2.convert("RGB")

# -----------------------------
# 분리수거 가이드 매핑
# -----------------------------

def map_guides(detections: List[Dict[str, Any]], guide_map: Dict[str, str]) -> List[Dict[str, Any]]:
    results = []
    for d in detections:
        cls = d.get("class")
        guide = guide_map.get(cls, "해당 분류에 대한 가이드가 없습니다.")
        results.append({**d, "guide": guide})
    return results

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="분리수거 보조 시스템", layout="wide")
    st.title("컴퓨터 비전 기반 분리수거 보조 시스템 (초안)")

    st.sidebar.header("설정")
    model_choice = st.sidebar.selectbox("모델 선택(설치된 경우)", options=["yolov8n.pt", "yolov8s.pt", "custom_model.pt"], index=0)
    model_path = st.sidebar.text_input("모델 파일 경로", value=model_choice)
    device = st.sidebar.selectbox("디바이스", options=["cpu", "cuda:0"], index=0)
    conf_thresh = st.sidebar.slider("신뢰도 임계값 (confidence)", min_value=0.0, max_value=1.0, value=0.25)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**분리수거 가이드(클래스명: 설명)**")
    guide_json = st.sidebar.text_area("가이드 매핑(JSON 형식)", value=json.dumps(DEFAULT_GUIDE, ensure_ascii=False, indent=2), height=200)

    # 가이드 파싱
    try:
        guide_map = json.loads(guide_json)
    except Exception:
        st.sidebar.error("가이드 JSON 파싱 실패. 기본 가이드를 사용합니다.")
        guide_map = DEFAULT_GUIDE

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 이미지 업로드")
        uploaded = st.file_uploader("이미지 파일 업로드 (.jpg/.png)", type=["jpg", "jpeg", "png"])
        use_camera = st.checkbox("카메라(웹캠)로 캡처 사용", value=False)

        if use_camera:
            img_file_buffer = st.camera_input("카메라로 촬영")
            if img_file_buffer is not None:
                uploaded = img_file_buffer

    with col2:
        st.markdown("### 모델 정보")
        if HAS_ULTRALYTICS:
            st.success("ultralytics 패키지 사용 가능: 실제 YOLO 모델로 예측합니다.")
        else:
            st.info("ultralytics 미설치: 더미 예측으로 동작합니다.")

        st.markdown("모델 파일 경로가 존재하지 않으면 더미 예측이 동작합니다.")

    if uploaded is None:
        st.info("이미지를 업로드하거나 카메라로 촬영해 주세요.")
        return

    # 이미지 읽기
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"이미지 읽기 실패: {e}")
        return

    st.markdown("### 원본 이미지")
    st.image(image, use_column_width=True)

    # 모델 로드(선택)
    with st.spinner("모델 로드 중..."):
        model = None
        if HAS_ULTRALYTICS and os.path.exists(model_path):
            model = load_yolo_model(model_path, device=device)
        else:
            if HAS_ULTRALYTICS and not os.path.exists(model_path):
                st.warning("모델 파일이 존재하지 않습니다. 더미 예측을 사용합니다.")

    # 예측
    with st.spinner("예측 수행 중..."):
        detections = predict_image_with_model(model, image, device=device, conf=conf_thresh)

    # 매핑 및 시각화
    mapped = map_guides(detections, guide_map)
    vis_img = draw_boxes(image, mapped)

    st.markdown("### 검출 결과 (시각화)")
    st.image(vis_img, use_column_width=True)

    st.markdown("### 검출된 항목 및 분리수거 가이드")
    if not mapped:
        st.write("감지된 항목이 없습니다.")
    else:
        for i, item in enumerate(mapped, 1):
            st.write(f"**{i}. 클래스:** {item['class']}  |  **신뢰도:** {item['score']:.2f}")
            st.write(f"**분리수거 가이드:** {item['guide']}")
            st.write(f"**바운딩박스:** {item['bbox']}")
            st.markdown("---")

    # 결과 JSON 다운로드
    result = {"detections": mapped, "meta": {"model": model_path, "conf_thresh": conf_thresh}}
    st.download_button("결과 JSON 다운로드", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="trash_detection_result.json", mime="application/json")

# -----------------------------
# 실행
# -----------------------------
if __name__ == '__main__':
    main()
