import cv2
import numpy as np

# 카메라 초기화
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Width 설정
cap.set(4, 240)  # Height 설정

# 트랙바 콜백 함수 (사용되지 않음)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Camera')

# 트랙바 생성
cv2.createTrackbar('Brightness', 'Camera', 40, 100, nothing)
cv2.createTrackbar('Contrast', 'Camera', 40, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera', 20, 100, nothing)

while True:
    # 트랙바 값 읽기
    brightness = cv2.getTrackbarPos('Brightness', 'Camera')
    contrast = cv2.getTrackbarPos('Contrast', 'Camera')
    saturation = cv2.getTrackbarPos('Saturation', 'Camera')
    gain = cv2.getTrackbarPos('Gain', 'Camera')

    # 카메라 속성 설정
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_GAIN, gain)

    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    # 프레임 표시
    cv2.imshow('Camera', frame)

    # 키 입력 대기 (30ms)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC 키를 누르면 종료
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
