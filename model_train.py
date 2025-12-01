from ultralytics import YOLO

def main():
    # 1) 사용할 모델 선택 (기본: yolov8n)
    model = YOLO("yolov8n.pt")  # COCO 사전학습 모델 기반 fine-tuning

    # 2) 훈련 실행
    model.train(
        data="datasets/recycle/data.yaml",   # 데이터셋 설정 파일
        epochs=100,                          # 학습 Epoch 수
        imgsz=640,                           # 입력 이미지 크기
        batch=16,
        device="cuda",                       # GPU 사용 (CPU는 "cpu")
        workers=4,
        name="recycle_train",                # 저장 폴더 runs/detect/recycle_train/
    )

    # 3) 검증(optional)
    model.val()

    print("학습이 완료되었습니다!")
    print("best.pt 위치:")
    print("   → runs/detect/recycle_train/weights/best.pt")

if __name__ == "__main__":
    main()
