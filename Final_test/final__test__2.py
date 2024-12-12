
import cv2
import numpy as np
import YB_Pcb_Car
import threading
import time
import os
import RPi.GPIO as GPIO
import random

# Initialize camera
def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # Set Width
    cap.set(4, 240)  # Set Height
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 25)
    cap.set(cv2.CAP_PROP_CONTRAST, 80)
    cap.set(cv2.CAP_PROP_SATURATION, 70)
    cap.set(cv2.CAP_PROP_GAIN, 80)
    return cap

cap = initialize_camera()
car = YB_Pcb_Car.YB_Pcb_Car()

# 트랙바 콜백 함수 (사용되지 않음)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Camera Settings')

# 트랙바 생성
cv2.createTrackbar('Servo 2 Angle', 'Camera Settings', 136, 180, nothing)
cv2.createTrackbar('Motor Up Speed', 'Camera Settings', 90, 125, nothing)
cv2.createTrackbar('R_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('G_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('B_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera Settings', 20, 100, nothing)

# 경고음 함수
def buzzer():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(32, GPIO.OUT)
    p = GPIO.PWM(32, 440)
    
    p.start(50)
    try:
        for dc in range(0, 70, 5):
            p.ChangeDutyCycle(dc)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        p.stop()
        GPIO.cleanup()

# 표지판 감지 및 제어 함수
def load_cascade(file_name):
    cascade = cv2.CascadeClassifier()
    if not cascade.load(cv2.samples.findFile(file_name)):
        print('--(!)Error loading cascade:', file_name)
        exit(0)
    return cascade

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print('--(!)Error capturing frame')
        return None
    return frame

def detect_object_sign(cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    object_sign = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return object_sign

def draw_rectangles_and_text(frame, object_sign):
    for (x, y, w, h) in object_sign:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Rectangle", (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame

def save_image(frame, count):
    os.makedirs("./save_images", exist_ok=True)
    file_name = f"./save_images/rect_{count}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Image saved: {file_name}")

# 자율 주행을 위한 프레임 처리 및 제어 함수
def weighted_gray(image, r_weight, g_weight, b_weight):
    r_weight /= 100.0
    g_weight /= 100.0
    b_weight /= 100.0
    return cv2.addWeighted(cv2.addWeighted(image[:, :, 2], r_weight, image[:, :, 1], g_weight, 0), 1.0, image[:, :, 0], b_weight, 0)

def process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value):
    pts_src = np.float32([[30, 60 + y_value], [280, 60 + y_value], [280, 10 + y_value], [30, 10 + y_value]])
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
    
    return binary_frame

def decide_direction(histogram, direction_threshold):
    length = len(histogram)
    left = int(np.sum(histogram[:length // 5]))
    right = int(np.sum(histogram[4 * length // 5:]))
    print("left:", left)
    print("right:", right)
    print("right - left:", right - left)
    if abs(right - left) > direction_threshold:
        return "LEFT" if right > left else "RIGHT"
    else:
        return "UP"

def control_car(direction, up_speed, down_speed):
    print(f"Controlling car: {direction}")
    if direction == "UP":
        car.Car_Run(up_speed - 40, up_speed - 40)
    elif direction == "LEFT":
        car.Car_Left(down_speed - 15, up_speed + 15)
    elif direction == "RIGHT":
        car.Car_Right(up_speed + 15, down_speed - 15)
    elif direction == "RANDOM":
        car.Car_Back(10,10)
        car.Car_Left(30,30)
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

# Load the cascade for stop sign detection
stop_cascade = load_cascade('cascade.xml')

try:
    while True:
        brightness = 23
        contrast = 80
        saturation = cv2.getTrackbarPos('Saturation', 'Camera Settings')
        gain = cv2.getTrackbarPos('Gain', 'Camera Settings')
        detect_value = 15
        motor_up_speed = 100
        motor_down_speed = 50
        #r_weight = cv2.getTrackbarPos('R_weight', 'Camera Settings')
        #g_weight = cv2.getTrackbarPos('G_weight', 'Camera Settings')
        #b_weight = cv2.getTrackbarPos('B_weight', 'Camera Settings')
        servo_1_angle = 90
        servo_2_angle = 132
        y_value = 10
        direction_threshold = 92000  # Adjusted threshold

        r_weight =33
        g_weight = 33
        b_weight = 33
        
        frame = capture_frame(cap)
        if frame is None:
            break

        rotate_servo(car, 1, servo_1_angle)
        rotate_servo(car, 2, servo_2_angle)

        processed_frame = process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value)
        histogram = np.sum(processed_frame, axis=0)
        print(f"Histogram: {histogram}")
        direction = decide_direction(histogram, direction_threshold)
        print(f"#### Decided direction ####: {direction}")
        cv2.imshow('4_Processed Frame', processed_frame)
        
        # Detect stop sign
        stop_signs = detect_object_sign(stop_cascade, frame)
        if len(stop_signs) > 0:
            print("Stop sign detected! Stopping car and sounding buzzer.")
            car.Car_Stop()
            buzzer()
        else:
            control_car(direction, motor_up_speed, motor_down_speed)

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
