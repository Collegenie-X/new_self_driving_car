import cv2
import time
import enum
import numpy as np
import YB_Pcb_Car  # Import Yahboom car library
import random
import RPi.GPIO as GPIO

######## 초음파 센서 셋팅
GPIO.setwarnings(False)

EchoPin = 18
TrigPin = 16

# Set GPIO port to BCM coding mode
GPIO.setmode(GPIO.BOARD)

GPIO.setup(EchoPin, GPIO.IN)
GPIO.setup(TrigPin, GPIO.OUT)

##### Buzzer Setting
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(32, GPIO.OUT)

p = GPIO.PWM(32, 220)

# Ultrasonic function
def Distance():
    GPIO.output(TrigPin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin, GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin, GPIO.LOW)

    t3 = time.time()

    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03:
            return -1
    t1 = time.time()
    while GPIO.input(EchoPin):
        t5 = time.time()
        if(t5 - t1) > 0.03:
            return -1

    t2 = time.time()
    time.sleep(0.01)
    return ((t2 - t1) * 340 / 2) * 100

def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
        distance = Distance()
        while int(distance) == -1:
            distance = Distance()
        while (int(distance) >= 500 or int(distance) == 0):
            distance = Distance()
        ultrasonic.append(distance)
        num += 1
        time.sleep(0.001)
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3]) / 3
    return distance

###### Set Camera
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # set Width
cap.set(4, 240)  # set Height

##### Set Motor
car = YB_Pcb_Car.YB_Pcb_Car()
MOTOR_UP_SPEED = 70  #### 65 0 ~ 125 Speed #### 70
MOTOR_DOWN_SPEED = 30  #### 40 #### 50

##### Set Brighter
DETECT_VALUE = 70

IS_STOP = False

def Up():
    car.Car_Run(MOTOR_UP_SPEED - 40, MOTOR_UP_SPEED - 40)
    time.sleep(0.1)
    if IS_STOP:
        car.Car_Stop()

def Down():
    car.Car_Back(MOTOR_UP_SPEED - 40, MOTOR_UP_SPEED - 40)
    time.sleep(0.1)
    if IS_STOP:
        car.Car_Stop()

def Left():
    car.Car_Left(MOTOR_DOWN_SPEED, MOTOR_UP_SPEED - 10)
    time.sleep(0.07)
    if IS_STOP:
        car.Car_Stop()

def Right():
    car.Car_Right(MOTOR_UP_SPEED - 10, MOTOR_DOWN_SPEED)
    time.sleep(0.07)
    if IS_STOP:
        car.Car_Stop()

def nothing(x):
    pass

# 트랙바 생성
cv2.namedWindow('Camera Settings')
cv2.createTrackbar('Servo 1 Angle', 'Camera Settings', 90, 180, nothing)
cv2.createTrackbar('Servo 2 Angle', 'Camera Settings', 113, 180, nothing)
cv2.createTrackbar('Y Value', 'Camera Settings', 10, 160, nothing)
cv2.createTrackbar('Direction Threshold', 'Camera Settings', 50000, 300000, nothing)
cv2.createTrackbar('Brightness', 'Camera Settings', 70, 100, nothing)
cv2.createTrackbar('Contrast', 'Camera Settings', 50, 100, nothing)
cv2.createTrackbar('Detect Value', 'Camera Settings', 70, 150, nothing)
cv2.createTrackbar('Motor Up Speed', 'Camera Settings', 90, 125, nothing)
cv2.createTrackbar('Motor Down Speed', 'Camera Settings', 50, 125, nothing)
cv2.createTrackbar('R_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('G_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('B_weight', 'Camera Settings', 33, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera Settings', 20, 100, nothing)
cv2.createTrackbar('Gain', 'Camera Settings', 20, 100, nothing)

try:
    while True:
        distance = Distance_test()
        if distance > 10:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            ####### 트랙바 값 읽기 ########
            servo_1_angle = cv2.getTrackbarPos('Servo 1 Angle', 'Camera Settings')
            servo_2_angle = cv2.getTrackbarPos('Servo 2 Angle', 'Camera Settings')
            y_value = cv2.getTrackbarPos('Y Value', 'Camera Settings')
            direction_threshold = cv2.getTrackbarPos('Direction Threshold', 'Camera Settings')
            brightness = cv2.getTrackbarPos('Brightness', 'Camera Settings')
            contrast = cv2.getTrackbarPos('Contrast', 'Camera Settings')
            detect_value = cv2.getTrackbarPos('Detect Value', 'Camera Settings')
            motor_up_speed = cv2.getTrackbarPos('Motor Up Speed', 'Camera Settings')
            motor_down_speed = cv2.getTrackbarPos('Motor Down Speed', 'Camera Settings')
            r_weight = cv2.getTrackbarPos('R_weight', 'Camera Settings')
            g_weight = cv2.getTrackbarPos('G_weight', 'Camera Settings')
            b_weight = cv2.getTrackbarPos('B_weight', 'Camera Settings')
            saturation = cv2.getTrackbarPos('Saturation', 'Camera Settings')
            gain = cv2.getTrackbarPos('Gain', 'Camera Settings')

            # 카메라 속성 설정
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            cap.set(cv2.CAP_PROP_SATURATION, saturation)
            cap.set(cv2.CAP_PROP_GAIN, gain)

            # 서보 모터 각도 설정
            car.Ctrl_Servo(1, servo_1_angle)
            car.Ctrl_Servo(2, servo_2_angle)

            ####### polylines ###########
            limited_polylines_list = [[10, 80 + y_value], [310, 80 + y_value], [310, 10 + y_value], [10, 10 + y_value]]
            limited_polylines_list_1 = [[8, 82 + y_value], [312, 82 + y_value], [312, 8 + y_value], [8, 8 + y_value]]
            pts = np.array(limited_polylines_list_1, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            cv2.imshow("1_polylines", frame)

            matSrc = np.float32(limited_polylines_list)
            matDst = np.float32([[0, 240], [320, 240], [320, 0], [0, 0]])
            matAffine = cv2.getPerspectiveTransform(matSrc, matDst)  # mat 1 src 2 dst
            limited_frame = cv2.warpPerspective(frame, matAffine, (320, 240))

            cv2.imshow("2_limited_frame", limited_frame)

            ######### limited frame (색깔 인식)
            limited_frame_copy = limited_frame.copy()

            hsv = cv2.cvtColor(limited_frame_copy, cv2.COLOR_BGR2HSV)
            hue, _, _ = cv2.split(hsv)
            mean_of_hue = cv2.mean(hue)[0]
            print(mean_of_hue)  #### 자신에게 맞는 색깔 찾기

            hue = cv2.inRange(hue, 160, 180)  ###### Red Mask
            red_frame = cv2.bitwise_and(hsv, hsv, mask=hue)
            red_frame = cv2.cvtColor(red_frame, cv2.COLOR_HSV2BGR)

            cv2.imshow("3_red_frame", red_frame)
            mean_of_hue = cv2.mean(hue)[0]
            print(mean_of_hue)

            if mean_of_hue > 10:
                p.start(20)  ### 소리 "삐익"
                car.Car_Stop()  ### 정지
                time.sleep(0.5)  ### 0.5초간 유지
                print("Red:", mean_of_hue)
                p.stop()  ### "삐익" 소리 중지

            ##### gray ############
            gray_frame = cv2.cvtColor(limited_frame, cv2.COLOR_RGB2GRAY)
            dst_retval, dst_binaryzation = cv2.threshold(gray_frame, detect_value, 255, cv2.THRESH_BINARY)  #### 밝기부분
            dst_binaryzation = cv2.erode(dst_binaryzation, None, iterations=1)

            cv2.imshow("4_gray_frame", gray_frame)
            cv2.imshow("5_dst_binaryzation", dst_binaryzation)
            print("dst_binaryzation", dst_binaryzation.shape)

            histogram = list(np.sum(dst_binaryzation[:, :], axis=0))  ##### 전체를 읽어서, 판단함.
            histogram_length = len(histogram)

            left = int(np.sum(histogram[:int(histogram_length / 4)]))
            right = int(np.sum(histogram[int(3 * histogram_length / 4):]))
            up = np.sum(histogram[int(2 * histogram_length / 4):int(3 * histogram_length / 4)])

            print("#####################")
            print("histogram", histogram)
            print("{}|--({})--|{} ".format(left, right - left, right))

            if abs(right - left) > direction_threshold:
                if right > left:  ### right 방향일 경우에...
                    print("[[ RIGHT ]]:", right - left)
                    Left()
                else:  #### Left 방향일 경우에
                    print("[[ LEFT ]]:", right - left)
                    Right()
            else:  #### Up(직진) 방향일 경우에....
                print("[[ UP ]]:", up)
                if up < 10000:
                    Up()
                else:
                    random_direction = random.randrange(1, 7)
                    car.Car_Stop()

        else:
            car.Car_Stop()
            print("----------------->An obstacle has been detected:", distance)
            p.start(20)
            car.Car_Right(20, 80)
            time.sleep(0.5)
            Up()
            time.sleep(0.5)
            Left()
            time.sleep(0.2)
            Up()
            time.sleep(0.5)
            p.stop()

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

except Exception as E:
    print("error:", E)
finally:
    car.Car_Stop()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
