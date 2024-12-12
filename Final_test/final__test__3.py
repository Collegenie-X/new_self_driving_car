#두원공과대학교 김재현 자율주행 2트랙

import cv2
import numpy as np
import YB_Pcb_Car
import random
import time

# 카메라와 자동차 초기화
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # set Width
cap.set(4, 240)  # set Height

car = YB_Pcb_Car.YB_Pcb_Car()

# 트랙바 콜백 함수 (사용되지 않음)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Camera Settings')

cv2.createTrackbar('Servo 2 Angle', 'Camera Settings', 136, 180, nothing)
cv2.createTrackbar('Motor Up Speed', 'Camera Settings', 90, 125, nothing)
cv2.createTrackbar('R_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('G_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('B_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera Settings', 20, 100, nothing)

def weighted_gray(image, r_weight, g_weight, b_weight):
    r_weight /= 100.0
    g_weight /= 100.0
    b_weight /= 100.0
    return cv2.addWeighted(cv2.addWeighted(image[:, :, 2], r_weight, image[:, :, 1], g_weight, 0), 1.0, image[:, :, 0], b_weight, 0)

def process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value):
    pts_src = np.float32([[10, 60 + y_value], [310, 60 + y_value], [310, 10 + y_value], [10, 10 + y_value]])
    pts_dst = np.float32([[0, 240], [320, 240], [320, 0], [0, 0]])

    pts = pts_src.reshape((-1, 1, 2)).astype(np.int32)
    frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imshow('1_Frame', frame)

    mat_affine = cv2.getPerspectiveTransform(pts_src, pts_dst)
    frame_transformed = cv2.warpPerspective(frame, mat_affine, (320, 240))
    cv2.imshow('2_frame_transformed', frame_transformed)

    gray_frame = weighted_gray(frame_transformed, r_weight, g_weight, b_weight)
    cv2.imshow('3_gray_frame', gray_frame)
    _, binary_frame = cv2.threshold(gray_frame, detect_value, 255, cv2.THRESH_BINARY)

    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
    binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)

    return binary_frame

def decide_direction(histogram, direction_threshold):
    length = len(histogram)
    left = int(np.sum(histogram[:length // 3]))
    center = int(np.sum(histogram[length // 3: 2 * length // 3]))
    right = int(np.sum(histogram[2 * length // 3:]))

    print("left:", left)
    print("center:", center)
    print("right:", right)

    if abs(right - left) > direction_threshold:
        return "LEFT" if right > left else "RIGHT"
    elif abs(center - left) > direction_threshold and abs(center - right) > direction_threshold:
        return "RANDOM"
    else:
        return "UP"

def control_car(direction, up_speed, down_speed):
    print(f"Controlling car: {direction}")
    if direction == "UP":
        car.Car_Run(up_speed - 40, up_speed - 40)
    elif direction == "LEFT":
        car.Car_Left(down_speed - 30, up_speed + 30)
    elif direction == "RIGHT":
        car.Car_Right(up_speed + 30, down_speed - 30)
    elif direction == "RANDOM":
        car.Car_Spin_Left(60, 60)
        for angle in range(70, 110, 10):
            rotate_servo(car, 1, angle)
            time.sleep(0.5)
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                continue
            processed_frame = process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value)
            histogram = np.sum(processed_frame, axis=0)
            direction = decide_direction(histogram, direction_threshold)
            if direction != "RANDOM":
                control_car(direction, up_speed, down_speed)
                break

def rotate_servo(car, servo_id, angle):
    car.Ctrl_Servo(servo_id, angle)

try:
    while True:
        brightness = 30
        contrast = 80
        saturation = cv2.getTrackbarPos('Saturation', 'Camera Settings')
        gain = cv2.getTrackbarPos('Gain', 'Camera Settings')
        detect_value = 15
        motor_up_speed = 105
        motor_down_speed = 50
        r_weight = cv2.getTrackbarPos('R_weight', 'Camera Settings')
        g_weight = cv2.getTrackbarPos('G_weight', 'Camera Settings')
        b_weight = cv2.getTrackbarPos('B_weight', 'Camera Settings')
        servo_1_angle = 90
        servo_2_angle = 130
        y_value = 10
        direction_threshold = 50000  # Adjusted threshold

        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        rotate_servo(car, 1, servo_1_angle)
        rotate_servo(car, 2, servo_2_angle)

        processed_frame = process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value)
        histogram = np.sum(processed_frame, axis=0)
        print(f"Histogram: {histogram}")
        direction = decide_direction(histogram, direction_threshold)
        print(f"#### Decided direction ####: {direction}")
        control_car(direction, motor_up_speed, motor_down_speed)

        cv2.imshow('4_Processed Frame', processed_frame)

        key = cv2.waitKey(30) & 0xff
        if key == 27:  # press 'ESC' to quit
            break
        elif key == 32:  # press 'Space bar' for pause and debug
            print("Paused for debugging. Press any key to continue.")
            cv2.waitKey()

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    car.Car_Stop()
    cap.release()
    cv2.destroyAllWindows()
