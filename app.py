from flask import Flask, render_template, Response, request
import cv2
import threading
import time
import pygame
import os
import RPi.GPIO as GPIO

app = Flask(__name__)

# GPIO setup
GPIO.setmode(GPIO.BCM)
TRIGGER_PIN = 23
ECHO_PIN = 24
SERVO_PIN_1 = 12
SERVO_PIN_2 = 16
SERVO_PIN_3 = 20

GPIO.setup(TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)
GPIO.setup(SERVO_PIN_3, GPIO.OUT)

servo1 = GPIO.PWM(SERVO_PIN_1, 50)  # 50Hz PWM frequency
servo2 = GPIO.PWM(SERVO_PIN_2, 50)
servo3 = GPIO.PWM(SERVO_PIN_3, 50)
servo1.start(7.5)  # Center position
servo2.start(7.5)  # Center position
servo3.start(7.5)  # Center position

def distance_measurement():
    # Measure distance in inches
    GPIO.output(TRIGGER_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_PIN, False)
    start_time = time.time()
    stop_time = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()
    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2
    distance_inches = distance / 2.54
    return round(distance_inches, 2)

# Set up Pygame for capturing key events
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
pygame.init()
pygame.display.set_caption('Distance Measurement')
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)

# Camera Class
class Camera:
    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.window_width = 640
        self.window_height = 480
        self.running = False
        self.text = ""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.thread = None
        self.start_time = None
        self.scroll_enabled = False
        self.overlay_enabled = True  # Start with the overlay enabled
        self.code_lines = [
            "import cv2",
            "import numpy as np",
            "cap = cv2.VideoCapture(0)",
            "while True:",
            "    ret, frame = cap.read()",
            "    if not ret:",
            "        break",
            "    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)",
            "    cv2.imshow('frame', gray)",
            "    if cv2.waitKey(1) & 0xFF == ord('q'):",
            "        break",
            "cap.release()",
            "cv2.destroyAllWindows()"
        ]
        self.scroll_index = 0

    def apply_overlay(self, frame):
        # Create a transparent overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Draw horizontal and vertical lines
        cv2.line(overlay, (0, height // 2), (width, height // 2), (0, 0, 255), 1)  # Red horizontal line
        cv2.line(overlay, (width // 2, 0), (width // 2, height), (0, 0, 255), 1)  # Red vertical line

        # Draw inch and centimeter marks
        inch_interval = 50  # Pixels per inch
        cm_interval = int(inch_interval * 2.54 / 1)  # Pixels per centimeter

        # Horizontal line marks
        for i in range(0, width // 2, inch_interval):
            if i != 0:
                # Inch marks
                cv2.line(overlay, (width // 2 + i, height // 2 - 5), (width // 2 + i, height // 2 + 5), (0, 0, 255), 1)
                cv2.line(overlay, (width // 2 - i, height // 2 - 5), (width // 2 - i, height // 2 + 5), (0, 0, 255), 1)
                # Centimeter marks
                cv2.line(overlay, (width // 2 + int(i * 2.54 / 1), height // 2 - 3), (width // 2 + int(i * 2.54 / 1), height // 2 + 3), (0, 255, 0), 1)
                cv2.line(overlay, (width // 2 - int(i * 2.54 / 1), height // 2 - 3), (width // 2 - int(i * 2.54 / 1), height // 2 + 3), (0, 255, 0), 1)

        # Vertical line marks
        for i in range(0, height // 2, inch_interval):
            if i != 0:
                # Inch marks
                cv2.line(overlay, (width // 2 - 5, height // 2 + i), (width // 2 + 5, height // 2 + i), (0, 0, 255), 1)
                cv2.line(overlay, (width // 2 - 5, height // 2 - i), (width // 2 + 5, height // 2 - i), (0, 0, 255), 1)
                # Centimeter marks
                cv2.line(overlay, (width // 2 - 3, height // 2 + int(i * 2.54 / 1)), (width // 2 + 3, height // 2 + int(i * 2.54 / 1)), (0, 255, 0), 1)
                cv2.line(overlay, (width // 2 - 3, height // 2 - int(i * 2.54 / 1)), (width // 2 + 3, height // 2 - int(i * 2.54 / 1)), (0, 255, 0), 1)

        # Combine the overlay with the frame
        cv2.addWeighted(overlay, 1, frame, 0.7, 0, frame)

        return frame

    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.running = False
            return

        self.start_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Rotate the frame by 180 degrees
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Resize the frame according to window size
            resized_frame = cv2.resize(rotated_frame, (self.window_width, self.window_height))

            # Apply overlay if enabled
            if self.overlay_enabled:
                resized_frame = self.apply_overlay(resized_frame)

            # Add text overlay after a 5-second delay
            if self.text and (time.time() - self.start_time) > 5:
                text_size = cv2.getTextSize(self.text, self.font, 1, 2)[0]
                text_x = self.window_width - text_size[0] - 10
                text_y = self.window_height - text_size[1] - 10
                cv2.putText(resized_frame, self.text, (text_x, text_y), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green text

            # Add scrolling code overlay if enabled
            if self.scroll_enabled:
                overlay_x = 10
                overlay_y = 20
                for i in range(5):  # Display up to 5 lines of code
                    if self.scroll_index + i < len(self.code_lines):
                        y = overlay_y + i * 15
                        line = self.code_lines[self.scroll_index + i]
                        cv2.putText(resized_frame, line, (overlay_x, y), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)  # Green text

                self.scroll_index = (self.scroll_index + 1) % len(self.code_lines)

            # Display the resized frame
            cv2.imshow('Logitech 4K Pro Webcam', resized_frame)

            # Check for key press events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('b'):  # Press 'b' to make the window larger
                self.window_width += 20
                self.window_height += 20
            elif key == ord('s'):  # Press 's' to make the window smaller
                self.window_width = max(20, self.window_width - 20)
                self.window_height = max(20, self.window_height - 20)
            elif key == ord('p'):  # Press 'p' to toggle the scrolling code
                self.scroll_enabled = not self.scroll_enabled
            elif key == ord('t'):  # Press 't' to toggle the overlay
                self.overlay_enabled = not self.overlay_enabled
                print(f"Overlay enabled: {self.overlay_enabled}")

            # Handle F keys for servo control
            if key == 0x70:  # F1 key
                servo3.ChangeDutyCycle(5)  # Power position
            elif key == 0x71:  # F2 key
                servo3.ChangeDutyCycle(10)  # Mode position
            elif key == 0x75:  # F6 key
                servo2.ChangeDutyCycle(10)  # Right position (maximum)
            elif key == 0x76:  # F7 key
                servo1.ChangeDutyCycle(10)  # Backward position (maximum)
            elif key == 0x77:  # F8 key
                servo1.ChangeDutyCycle(5)  # Forward position (minimum)
            elif key == 0x78:  # F9 key
                servo2.ChangeDutyCycle(5)  # Left position (minimum)
            elif key == ord('c'):
                self.stop()  # Stop the camera
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()
            self.thread = None
            cv2.destroyAllWindows()

# Initialize the camera
camera = Camera(index=0)

def generate_frames():
    while True:
        success, frame = camera.cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control/<key>')
def control(key):
    try:
        if key == 'f1':
            servo3.ChangeDutyCycle(5)  # Power position
        elif key == 'f2':
            servo3.ChangeDutyCycle(10)  # Mode position
        elif key == 'f6':
            servo2.ChangeDutyCycle(10)  # Right position (maximum)
        elif key == 'f7':
            servo1.ChangeDutyCycle(10)  # Backward position (maximum)
        elif key == 'f8':
            servo1.ChangeDutyCycle(5)  # Forward position (minimum)
        elif key == 'f9':
            servo2.ChangeDutyCycle(5)  # Left position (minimum)
        elif key == 'q':
            global running
            running = False
        elif key == 'stop':
            # Reset servos to center position
            servo1.ChangeDutyCycle(7.5)
            servo2.ChangeDutyCycle(7.5)
            servo3.ChangeDutyCycle(7.5)
        elif key == 'c':
            if camera.running:
                camera.stop()
            else:
                camera.start()
        elif key == 'p':
            camera.scroll_enabled = not camera.scroll_enabled
        elif key == 't':
            camera.overlay_enabled = not camera.overlay_enabled
            print(f"Overlay enabled: {camera.overlay_enabled}")
        return ('', 204)
    except Exception as e:
        print(f"Error in control route: {e}")
        return "Error in control route", 500

if __name__ == "__main__":
    camera.start()
    app.run(host='0.0.0.0', port=5001, debug=True)
