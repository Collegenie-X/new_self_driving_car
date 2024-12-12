import cv2
import numpy as np
import YB_Pcb_Car
import random
import time
import RPi.GPIO as GPIO

car = YB_Pcb_Car.YB_Pcb_Car()

# 소리를 내는 함수
def buzzer():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(32, GPIO.OUT)
    p = GPIO.PWM(32, 440)
    p.start(50)
    try:
        for dc in range(0, 50, 5):
            p.ChangeDutyCycle(dc)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        p.stop()
        GPIO.cleanup()

# 카메라 불러옴
def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # Set Width
    cap.set(4, 240)  # Set Height
    return cap

# cascade 불러옴
def load_cascade(file_name):
    cascade = cv2.CascadeClassifier()
    if not cascade.load(cv2.samples.findFile(file_name)):
        print('--(!)Error loading cascade:', file_name)
        exit(0)
    return cascade

# sign
def detect_object_sign(cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    object_signs = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return object_signs


def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print('--(!)Error capturing frame')
        return None
    return frame

# 트랙바 콜백 함수 (사용되지 않음)
def nothing(x):
    pass

# 윈도우 생성 (값을 조절하는 부분)
cv2.namedWindow('Camera Settings')

# 트랙바 생성
cv2.createTrackbar('Servo 1 Angle', 'Camera Settings', 107, 180, nothing)
cv2.createTrackbar('Servo 2 Angle', 'Camera Settings', 122, 180, nothing)
cv2.createTrackbar('Y Value', 'Camera Settings', 60, 160, nothing)
cv2.createTrackbar('Direction Threshold', 'Camera Settings', 60000, 300000, nothing)
cv2.createTrackbar('Up Threshold', 'Camera Settings', 50000, 300000, nothing)
cv2.createTrackbar('Brightness', 'Camera Settings', 30, 100, nothing)
cv2.createTrackbar('Contrast', 'Camera Settings', 70, 100, nothing)
cv2.createTrackbar('Detect Value', 'Camera Settings', 30, 150, nothing)
cv2.createTrackbar('Motor Up Speed', 'Camera Settings', 50, 125, nothing)
cv2.createTrackbar('Motor Down Speed', 'Camera Settings', 50, 125, nothing)
cv2.createTrackbar('R_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('G_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('B_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera Settings', 20, 100, nothing)

def weighted_gray(image, r_weight, g_weight, b_weight):
    # 가중치를 0-1 범위로 변환
    r_weight /= 100.0
    g_weight /= 100.0
    b_weight /= 100.0
    return cv2.addWeighted(cv2.addWeighted(image[:, :, 2], r_weight, image[:, :, 1], g_weight, 0), 1.0, image[:, :, 0], b_weight, 0)

def process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value):
    # Define region for perspective transformation
    pts_src = np.float32([[10, 70 + y_value], [310, 70 + y_value], [310, 10 + y_value], [10, 10 + y_value]])
    pts_dst = np.float32([[0, 240], [320, 240], [320, 0], [0, 0]])
    pts = pts_src.reshape((-1, 1, 2)).astype(np.int32)  # np.float32에서 np.int32로 변경
    frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    #cv2.imshow('1_Frame', frame)
    # Apply perspective transformation
    mat_affine = cv2.getPerspectiveTransform(pts_src, pts_dst)
    frame_transformed = cv2.warpPerspective(frame, mat_affine, (320, 240))
    #cv2.imshow('2_frame_transformed', frame_transformed)
    # Convert to grayscale using weighted gray
    gray_frame = weighted_gray(frame_transformed, r_weight, g_weight, b_weight)
    #cv2.imshow('3_gray_frame', gray_frame)
    _, binary_frame = cv2.threshold(gray_frame, detect_value, 255, cv2.THRESH_BINARY)
    return binary_frame

def decide_direction(histogram, direction_threshold, car, detect_value):
    length = len(histogram)
    left = int(np.sum(histogram[:length // 5]))
    right = int(np.sum(histogram[4 * length // 5:]))
    center = int(np.sum(histogram[2 * length // 5:4 * length // 5]))
    print("left:", left)
    print("right:", right)
    print("right - left:", right - left)
    print("center:", center)
    if abs(right - left) > direction_threshold:
        return "LEFT" if right > left else "RIGHT"
    else:
        ## 라인 바가 직진으로 LEFT/RIGHT가 구별되지 않는 경우
        if (center > 650000) : 
            
            car.Car_Stop()

            return rotate_servo_and_check_direction(car, detect_value, 33, 33, 33, 10, direction_threshold)
        else :
            return "UP"
def control_car(direction, up_speed, down_speed):
    print(f"Controlling car: {direction}")
    if direction == "UP":
        car.Car_Run(up_speed, up_speed)
    elif direction == "LEFT":
        car.Car_Left(0, up_speed+40)
    elif direction == "RIGHT":
        car.Car_Right(up_speed+40, 0)
    
def rotate_servo(car, servo_id, angle):
    car.Ctrl_Servo(servo_id, angle)

def rotate_servo_and_check_direction(car, detect_value, r_weight, g_weight, b_weight, y_value, direction_threshold):
    car.Ctrl_Servo(1, 180)
    time.sleep(0.5)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        return "STOP"
    processed_frame = process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value)
    histogram_180 = np.sum(processed_frame, axis=0)
    length = len(histogram_180)
    center = int(np.sum(histogram_180[3 * length // 5:4 * length // 5]))
    car.Ctrl_Servo(1, 90)
    time.sleep(0.5)
    if center > 5000:
        return "LEFT"
    else:
        return "RIGHT"
    
cap = initialize_camera()
object_cascade = load_cascade('/home/pi/Desktop/xml/cascade_Box.xml')



try:
    while True:
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
        up_threshold = cv2.getTrackbarPos('Up Threshold', 'Camera Settings')

        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_GAIN, gain)

        rotate_servo(car, 1, servo_1_angle)
        rotate_servo(car, 2, servo_2_angle)

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        
        object_signs = detect_object_sign(object_cascade, frame)
        if len(object_signs) > 0:
            car.Car_Stop()
            time.sleep(1)
            buzzer()
            time.sleep(3)
        else:
            time.sleep(0.1)

        
        processed_frame = process_frame(frame, detect_value, r_weight, g_weight, b_weight, y_value)
        histogram = np.sum(processed_frame, axis=0)

        print(f"Histogram: {histogram}")
        direction = decide_direction(histogram, direction_threshold, car, detect_value)
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
