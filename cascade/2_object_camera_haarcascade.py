import cv2
import numpy as np
import time
import os

def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)  # Set Width
    cap.set(4, 240)  # Set Height
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 70)
    cap.set(cv2.CAP_PROP_CONTRAST, 70)
    cap.set(cv2.CAP_PROP_SATURATION, 70)
    cap.set(cv2.CAP_PROP_GAIN, 80)
    return cap

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

def draw_rectangles_and_text(frame, traffic_sign):
    for (x, y, w, h) in traffic_sign:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, "Red Traffic Sign", (x - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame

def save_image(frame, count):
    os.makedirs("./save_images", exist_ok=True)
    file_name = f"./save_images/traffic_{count}.jpg"
    cv2.imwrite(file_name, frame)
    print(f"Image saved: {file_name}")

def main():
    cap = initialize_camera()
    object_cascade = load_cascade('cascade.xml')
    count = 0

    while True:
        frame = capture_frame(cap)
        if frame is None:
            continue
        
        object_sign = detect_object_sign(object_cascade, frame)

        if len(object_sign) > 0:
            frame = draw_rectangles_and_text(frame, object_sign)

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break
        if k == 32:  # press 'SPACE' to save image
            save_image(frame, count)
            count += 1

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
