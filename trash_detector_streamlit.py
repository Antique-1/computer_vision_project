import streamlit as st
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
from ultralytics import YOLO  

# -----------------------------
# 기본 분리수거 가이드 매핑
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
# 내가 학습한 모델 로드
# -----------------------------
MODEL_PATH = os.path.abspath("runs/train_auto/exp/weights/best.pt")     # ← 사용자가 학습한 모델 파일
MODEL = YOLO(MODEL_PATH)


# -----------------------------
# 예측 함수
# -----------------------------
def predict_image(img: Image.Image, device: str = "cpu", conf: float = 0.25) -> List[Dict[str, Any]]:
    arr = np.array(img.convert("RGB"))
    results = MODEL(arr, conf=conf, device=device)

    out = []
    r = results[0]
    boxes = r.boxes
    cls_names = MODEL.names

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = cls_names.get(cls_id, str(cls_id))

        out.append({
            "class": cls_name,
            "score": score,
            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
        })

    return out


# -----------------------------
# 시각화 함수
# -----------------------------
def draw_boxes(img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    img2 = img.convert("RGBA")
    draw = ImageDraw.Draw(img2)
    font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        score = det["score"]

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)

        text = f"{cls} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=(255, 0, 0, 200))
        draw.text((x1 + 2, y1 - text_h - 2), text, fill=(255, 255, 255, 255), font=font)

    return img2.convert("RGB")


# -----------------------------
# 가이드 매핑
# -----------------------------
def map_guides(detections: List[Dict[str, Any]], guide_map: Dict[str, str]):
    results = []
    for d in detections:
        guide = guide_map.get(d["class"], "해당 분류에 대한 가이드가 없습니다.")
        results.append({**d, "guide": guide})
    return results


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="분리수거 보조 시스템", layout="wide")
    st.title("컴퓨터 비전 기반 분리수거 보조 시스템")

    st.sidebar.header("설정")
    device = st.sidebar.selectbox("디바이스", ["cpu", "cuda:0"], index=0)
    conf_thresh = st.sidebar.slider("신뢰도 임계값", 0.0, 1.0, 0.25)

    st.sidebar.markdown("**분리수거 가이드 JSON**")
    guide_json = st.sidebar.text_area(
        "가이드 매핑(JSON 형식)",
        value=json.dumps(DEFAULT_GUIDE, ensure_ascii=False, indent=2),
        height=200
    )

    try:
        guide_map = json.loads(guide_json)
    except:
        st.sidebar.error("JSON 파싱 실패 → 기본 가이드 사용")
        guide_map = DEFAULT_GUIDE

    uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("이미지를 업로드해 주세요.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.markdown("### 원본 이미지")
    st.image(image, use_column_width=True)

    with st.spinner("예측 중..."):
        detections = predict_image(image, device=device, conf=conf_thresh)

    mapped = map_guides(detections, guide_map)
    vis_img = draw_boxes(image, mapped)

    st.markdown("### 검출 결과")
    st.image(vis_img, use_column_width=True)

    for i, item in enumerate(mapped, 1):
        st.write(f"**{i}. 클래스:** {item['class']} | **신뢰도:** {item['score']:.2f}")
        st.write(f"**분리수거 가이드:** {item['guide']}")
        st.write(f"**바운딩박스:** {item['bbox']}")
        st.markdown("---")


if __name__ == '__main__':
    main()
