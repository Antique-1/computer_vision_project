# 컴퓨터 비전 프로젝트

## :pencil: 팀원 구성

* 20181510 윤지환 - ...
* 20201710 박광우 - ...
* 20211701 김도현 - ...

## :mag_right: 프로젝트 주제 
* 손 포즈 탐지
* 사람 혹은 객체의 수 탐지
## :notebook_with_decorative_cover: 참고 프로젝트 및 문서
* [경진용 데이터셋을 사용해서 YOLOv5로 커스텀 객체 탐지 모델을 만드는 튜토리얼](https://readmedium.com/yolov5-tutorial-on-custom-object-detection-using-kaggle-competition-dataset-1ff76219d82a?utm_source=chatgpt.com)
  + 해당 튜토리얼은 데이터 전처리 → 모델 학습 → 추론 → 결과 시각화까지, 객체 탐지 프로젝트 전 과정을 단계별로 설명하고 있어 초보자도 따라하기 쉬워서 참고 문서로 선택하였음
## :high_brightness: 컴퓨터 비전 활용 요소 분석

### 📌 1. 객체 탐지 (Object Detection)

- 이미지 내에서 객체의 **위치(Bounding Box)**와 **종류(Class)**를 동시에 예측하는 기술.
- 본 프로젝트에서는 **YOLOv5**를 사용하여 단일 프레임에서 다수의 객체를 실시간으로 탐지함.
- YOLO는 **One-Stage Detector**로, 속도와 정확도 모두 우수함.

---

### 📌 2. 전이 학습 (Transfer Learning)

- 대규모 데이터셋(COCO 등)으로 사전 학습된 YOLOv5 모델을 기반으로, 커스텀 데이터셋에 재학습을 수행함.
- 이를 통해 **적은 데이터로도 좋은 성능을 낼 수 있으며**, 학습 시간도 단축됨.

---

---


## :bulb: 참고하는 이유
