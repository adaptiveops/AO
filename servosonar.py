#!/usr/bin/env python3
import time
import pygame
from gpiozero import DistanceSensor, Servo
import cv2
import threading
import os
import math
import random
from datetime import datetime

# Constants for window size (in pixels, assuming 96 DPI)
DATA_BOX_WIDTH = int(2 * 96)
DATA_BOX_HEIGHT = int(3 * 96)

############################# Distance Sensor Setup #############################
sensor = DistanceSensor(echo=24, trigger=23)

def distance_measurement():
    return round(sensor.distance * 100 / 2.54, 2)  # Distance in inches

############################# Servo Setup #############################
servos = [Servo(pin, min_pulse_width=1/1000, max_pulse_width=2/1000) for pin in (12, 16, 20)]

############################# Pygame Setup #############################
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
pygame.init()
pygame.display.set_caption('Distance Measurement and Servo Control')
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)
pygame.mixer.init()

sound_paths = {
    pygame.K_F1: 'Robot Voice - power.wav',
    pygame.K_F2: 'Robot Voice - sequence.wav',
    pygame.K_F4: 'Robot Voice - autopilot.wav',
    pygame.K_F6: 'Robot Voice - left.wav',
    pygame.K_F7: 'Robot Voice - accelerate.wav',
    pygame.K_F8: 'Robot Voice - warning.wav',
    pygame.K_F9: 'Robot Voice - right.wav',
    pygame.K_q: 'Robot Voice - Disabled.wav',
    'startup': 'Robot Voice - warning.wav',
    'data': 'Robot Voice - data.wav',
    'nightvision': '/home/aounit1/Desktop/AO/AO Sounds/nightvision.wav'  # New sound for overlay
}
sounds = {k: pygame.mixer.Sound(f'/home/aounit1/Desktop/AO/AO Sounds/{v}') for k, v in sound_paths.items() if k != 'nightvision'}
sounds['nightvision'] = pygame.mixer.Sound(sound_paths['nightvision'])

for _ in range(3):
    sounds['startup'].play()
    time.sleep(1.5)

############################# Camera Setup #############################
class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        self.window_size = (640, 480)
        self.running = False
        self.overlay_enabled = False
        self.night_vision_enabled = False
        self.scroll_enabled = False
        self.data_window_enabled = False
        self.face_recognition_enabled = False
        self.text = ""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.code_lines = [
            "import cv2", "import numpy as np", "cap = cv2.VideoCapture(0)", "while True:",
            "    ret, frame = cap.read()", "    if not ret:", "        break",
            "    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", "    cv2.imshow('frame', gray)",
            "    if cv2.waitKey(1) & 0xFF == ord('q'):", "        break",
            "cap.release()", "cv2.destroyAllWindows()"
        ]
        self.scroll_index = 0
        # Update the path to the Haar cascade file
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def apply_overlay(self, frame):
        overlay = frame.copy()
        height, width = frame.shape[:2]
        for i in range(0, width // 4, 10):
            if i != 0:
                cv2.line(overlay, (width // 2 + i, height // 2 - 3), (width // 2 + i, height // 2 + 3), (0, 255, 0), 1)
                cv2.line(overlay, (width // 2 - i, height // 2 - 3), (width // 2 - i, height // 2 + 3), (0, 255, 0), 1)
        for i in range(0, height // 4, 10):
            if i != 0:
                cv2.line(overlay, (width // 2 - 3, height // 2 + i), (width // 2 + 3, height // 2 + i), (0, 255, 0), 1)
                cv2.line(overlay, (width // 2 - 3, height // 2 - i), (width // 2 + 3, height // 2 - i), (0, 255, 0), 1)
        cv2.addWeighted(overlay, 1, frame, 0.7, 0, frame)
        return frame

    def draw_data_window(self, frame):
        x_offset, y_offset = 10, 10
        for i in range(5):
            line = self.code_lines[(self.scroll_index + i) % len(self.code_lines)]
            cv2.putText(frame, line, (x_offset, y_offset + i * 20), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Sonar: {random.uniform(1.0, 10.0):.2f} in", (x_offset, y_offset + 5 * 20), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (x_offset, y_offset + 6 * 20), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        self.scroll_index = (self.scroll_index + 1) % len(self.code_lines)
        return frame

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

    def run(self):
        start_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if self.night_vision_enabled:
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                frame_hsv[:, :, 2] = cv2.equalizeHist(frame_hsv[:, :, 2])
                frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
            frame = cv2.resize(frame, self.window_size)
            if self.face_recognition_enabled:
                frame = self.detect_faces(frame)
            if self.overlay_enabled:
                frame = self.apply_overlay(frame)
            if self.data_window_enabled:
                frame = self.draw_data_window(frame)
            if self.text and (time.time() - start_time) > 5:
                text_size = cv2.getTextSize(self.text, self.font, 1, 2)[0]
                text_x, text_y = self.window_size[0] - text_size[0] - 10, self.window_size[1] - text_size[1] - 10
                cv2.putText(frame, self.text, (text_x, text_y), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if self.scroll_enabled:
                for i in range(5):
                    if self.scroll_index + i < len(self.code_lines):
                        y = 20 + i * 15
                        cv2.putText(frame, self.code_lines[self.scroll_index + i], (10, y), self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                self.scroll_index = (self.scroll_index + 1) % len(self.code_lines)
            cv2.imshow('Logitech 4K Pro Webcam', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('b'):
                self.window_size = (self.window_size[0] + 20, self.window_size[1] + 20)
            elif key == ord('s'):
                self.window_size = (max(20, self.window_size[0] - 20), max(20, self.window_size[1] - 20))
            elif key == ord('p'):
                if not self.data_window_enabled:
                    sounds['data'].play()
                self.data_window_enabled = not self.data_window_enabled
            elif key == ord('t'):
                if not self.overlay_enabled:
                    sounds['nightvision'].play()  # Play sound when overlay is enabled
                self.overlay_enabled = not self.overlay_enabled
            elif key == ord('f'):
                self.face_recognition_enabled = not self.face_recognition_enabled
            elif key == ord('w'):
                self.cap.set(cv2.CAP_PROP_AUTO_WB, not self.cap.get(cv2.CAP_PROP_AUTO_WB))
            elif key == ord('n'):
                self.night_vision_enabled = not self.night_vision_enabled
            elif key == ord('c'):
                self.stop()
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.run).start()

    def stop(self):
        if self.running:
            self.running = False
            cv2.destroyAllWindows()

# Initialize the camera
camera = Camera(index=0)
camera.start()

############################# Continuous Servo Movement #############################
continuous_servo = False
running = True
override_servos = False

def continuous_servo_movement():
    global continuous_servo, override_servos
    while True:
        if continuous_servo and not override_servos:
            for i in range(360):
                if not continuous_servo or override_servos:
                    break
                servos[0].value = math.sin(math.radians(i))
                servos[1].value = math.cos(math.radians(i))
                servos[2].value = (servos[0].value + servos[1].value) / 2
                time.sleep(0.01)
        time.sleep(0.1)

threading.Thread(target=continuous_servo_movement, daemon=True).start()

# Delay before starting servos
time.sleep(2)

############################# Main Loop #############################
try:
    while running:
        distance = distance_measurement()
        print("Distance:", distance)
        camera.text = f"Distance: {distance} inches"

        # Check the distance and override servos if an object is within 5 inches
        if distance < 5:
            if not override_servos:
                for servo in servos:
                    servo.value = 0  # Center position
                override_servos = True
        else:
            if override_servos:
                override_servos = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                servos[0].value = servos[1].value = servos[2].value = 0
                sounds[pygame.K_q].play()
                time.sleep(2)
                running = False
            elif event.type == pygame.KEYDOWN:
                if not override_servos and event.key in sounds:
                    if event.key == pygame.K_F8:
                        sounds[pygame.K_F8].play()
                    elif event.key == pygame.K_F4:
                        continuous_servo = not continuous_servo
                        sounds[event.key].play() if continuous_servo else None
                    else:
                        sounds[event.key].play()

        if not override_servos:
            keys = pygame.key.get_pressed()
            servos[0].value = 1 if keys[pygame.K_F7] else -1 if keys[pygame.K_F8] else 0
            servos[1].value = 1 if keys[pygame.K_F6] else -1 if keys[pygame.K_F9] else 0
            servos[2].value = 1 if keys[pygame.K_F2] else -1 if keys[pygame.K_F1] else 0

        pygame.display.update()
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    if camera.running:
        camera.stop()
    pygame.quit()
