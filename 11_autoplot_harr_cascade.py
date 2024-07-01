import cv2
import numpy as np
import YB_Pcb_Car
import threading
import random
import time

# Camera and car initialization
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Set width
cap.set(4, 240)  # Set height

car = YB_Pcb_Car.YB_Pcb_Car()

# 트랙바 콜백 함수 (사용되지 않음)
def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Camera Settings')

# 트랙바 생성
cv2.createTrackbar('Servo 1 Angle', 'Camera Settings', 90, 180, nothing)
cv2.createTrackbar('Servo 2 Angle', 'Camera Settings', 113, 180, nothing)
cv2.createTrackbar('Y Value', 'Camera Settings', 10, 160, nothing)
cv2.createTrackbar('Direction Threshold', 'Camera Settings', 50000, 300000, nothing)
cv2.createTrackbar('Brightness', 'Camera Settings', 65, 100, nothing)
cv2.createTrackbar('Contrast', 'Camera Settings', 80, 100, nothing)
cv2.createTrackbar('Detect Value', 'Camera Settings', 15, 150, nothing)
cv2.createTrackbar('Motor Up Speed', 'Camera Settings', 90, 125, nothing)
cv2.createTrackbar('Motor Down Speed', 'Camera Settings', 50, 125, nothing)
cv2.createTrackbar('R_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('G_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('B_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera Settings', 20, 100, nothing)

# Haar Cascade models 경로 설정
obstacle_cascade_path = 'path_to_obstacle_cascade.xml'
traffic_light_cascade_path = 'path_to_traffic_light_cascade.xml'
sign_cascade_path = 'path_to_sign_cascade.xml'

# Haar Cascade models 로드
obstacle_cascade = cv2.CascadeClassifier(obstacle_cascade_path)
traffic_light_cascade = cv2.CascadeClassifier(traffic_light_cascade_path)
sign_cascade = cv2.CascadeClassifier(sign_cascade_path)

def weighted_gray(image, r_weight, g_weight, b_weight):
    # 가중치를 0-1 범위로 변환
    r_weight /= 100.0
    g_weight /= 100.0
    b_weight /= 100.0
    return cv2.addWeighted(cv2.addWeighted(image[:, :, 2], r_weight, image[:, :, 1], g_weight, 0), 1.0, image[:, :, 0], b_weight, 0)

def process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value):
    """
    Process the frame to detect edges and transform perspective.
    """
    # Define region for perspective transformation
    pts_src = np.float32([[10, 60 + y_value], [310, 60 + y_value], [310, 10 + y_value], [10, 10 + y_value]])
    pts_dst = np.float32([[0, 240], [320, 240], [320, 0], [0, 0]])

    # 사각형 그리기
    pts = pts_src.reshape((-1, 1, 2)).astype(np.int32)  # np.float32에서 np.int32로 변경
    frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imshow('1_Frame', frame)

    # Apply perspective transformation
    mat_affine = cv2.getPerspectiveTransform(pts_src, pts_dst)
    frame_transformed = cv2.warpPerspective(frame, mat_affine, (320, 240))
    cv2.imshow('2_frame_transformed', frame_transformed)

    # Convert to grayscale using weighted gray
    gray_frame = weighted_gray(frame_transformed, r_weight, g_weight, b_weight)
    cv2.imshow('3_gray_frame', gray_frame)
    _, binary_frame = cv2.threshold(gray_frame, detect_value, 255, cv2.THRESH_BINARY)
    return binary_frame

def decide_direction(histogram, direction_threshold):
    """
    Decide the driving direction based on histogram.
    """
    # 히스토그램의 길이
    length = len(histogram)

    # 히스토그램을 세 구역으로 나눔
    left = int(np.sum(histogram[:length // 5]))
    right = int(np.sum(histogram[4 * length // 5:]))

    print("left:", left)
    print("right:", right)
    print("right - left:", right - left)

    # 방향 결정
    if abs(right - left) > direction_threshold:
        return "LEFT" if right > left else "RIGHT"
    else:
        return "UP"

def control_car(direction, up_speed, down_speed):
    """
    Control the car based on the decided direction.
    """
    print(f"Controlling car: {direction}")
    if direction == "UP":
        car.Car_Run(up_speed - 35, up_speed - 35)
    elif direction == "LEFT":
        car.Car_Left(down_speed, up_speed)
    elif direction == "RIGHT":
        car.Car_Right(up_speed, down_speed)
    elif direction == "RANDOM":
        random_direction = random.choice(["LEFT", "RIGHT"])
        control_car(random_direction, up_speed, down_speed)

def rotate_servo(car, servo_id, angle):
    car.Ctrl_Servo(servo_id, angle)

def detect_obstacle(frame, control_signals):
    if obstacle_cascade.empty():
        print("Obstacle cascade not loaded.")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    obstacles = obstacle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in obstacles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    control_signals['obstacle'] = len(obstacles) > 0

def detect_traffic_light(frame, control_signals):
    if traffic_light_cascade.empty():
        print("Traffic light cascade not loaded.")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    traffic_lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in traffic_lights:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    control_signals['red_light'] = len(traffic_lights) > 0

def detect_sign(frame, control_signals):
    if sign_cascade.empty():
        print("Sign cascade not loaded.")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signs = sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in signs:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Assuming the first detected sign is the one we're interested in
    control_signals['sign'] = 'O' if signs else 'X'

try:
    while True:
        # 트랙바 값 읽기
        brightness = cv2.getTrackbarPos('Brightness', 'Camera Settings')
        contrast = cv2.getTrackbarPos('Contrast', 'Camera Settings')
        saturation = cv2.getTrackbarPos('Saturation', 'Camera Settings')
        gain = cv2.getTrackbarPos('Gain', 'Camera Settings')
        detect_value = cv2.getTrackbarPos('Detect Value', 'Camera Settings')
        motor_up_speed = cv2.getTrackbarPos('Motor Up Speed', 'Camera Settings')
        motor_down_speed = cv2.getTrackbarPos('Motor Down Speed', 'Camera Settings')
        r_weight = cv2.getTrackbarPos('R_weight', 'Camera Settings')
        g_weight = cv2.getTrackbarPos('G_weight', 'Camera Settings')
        b_weight = cv2.getTrackbarPos('B_weight', 'Camera Settings')
        servo_1_angle = cv2.getTrackbarPos('Servo 1 Angle', 'Camera Settings')
        servo_2_angle = cv2.getTrackbarPos('Servo 2 Angle', 'Camera Settings')
        y_value = cv2.getTrackbarPos('Y Value', 'Camera Settings')
        direction_threshold = cv2.getTrackbarPos('Direction Threshold', 'Camera Settings')

        # 카메라 속성 설정
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_GAIN, gain)
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # 서보 모터 각도 조절
        rotate_servo(car, 1, servo_1_angle)
        rotate_servo(car, 2, servo_2_angle)

        processed_frame = process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value)
        histogram = np.sum(processed_frame, axis=0)
        print(f"Histogram: {histogram}")
        direction = decide_direction(histogram, direction_threshold)
        print(f"#### Decided direction ####: {direction}")
        control_car(direction, motor_up_speed, motor_down_speed)

        # Display the processed frame (for debugging)
        cv2.imshow('4_Processed Frame', processed_frame)

        # Shared control signals dictionary
        control_signals = {'obstacle': False, 'red_light': False, 'sign': None}

        # Create and start threads for detection tasks
        obstacle_thread = threading.Thread(target=detect_obstacle, args=(frame, control_signals))
        traffic_light_thread = threading.Thread(target=detect_traffic_light, args=(frame, control_signals))
        sign_thread = threading.Thread(target=detect_sign, args=(frame, control_signals))

        obstacle_thread.start()
        traffic_light_thread.start()
        sign_thread.start()

        # Wait for threads to finish
        obstacle_thread.join()
        traffic_light_thread.join()
        sign_thread.join()

        # Autonomous driving logic based on detections
        if control_signals['obstacle']:
            print("Obstacle detected! Avoiding...")
            control_car('LEFT')  # Change to your obstacle avoidance strategy
        elif control_signals['red_light']:
            print("Red light detected! Stopping...")
            car.Car_Stop()  # Stop the car
        elif control_signals['sign'] == 'O':
            print("Sign 'O' detected! Parking...")
            car.Car_Stop()  # Implement your parking strategy

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
