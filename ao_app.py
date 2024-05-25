from gpiozero import Device, DistanceSensor, Servo
from gpiozero.pins.rpigpio import RPiGPIOFactory
import cv2
import pygame

# Set pin factory to RPiGPIOFactory
Device.pin_factory = RPiGPIOFactory()

# Initialize distance sensor
sensor = DistanceSensor(echo=24, trigger=23)


def distance_measurement():
    return round(sensor.distance * 100 / 2.54, 2)  # Distance in inches

############################# Pygame Setup #############################
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
pygame.init()
pygame.display.set_caption('Distance Measurement')
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)
sound_path = '/home/aounit1/Desktop/AO/sound-effect-night-vision-2_130bpm_C#.wav'
startup_sound_path = '/home/aounit1/Desktop/AO/AO Sounds/Robotic Voice Three Two One Go.mp3'
shutdown_sound_path = '/home/aounit1/Desktop/AO/AO Sounds/MusicAccent EC02_33_4.wav'
pygame.mixer.init()
sound_effect = pygame.mixer.Sound(sound_path)
startup_sound_effect = pygame.mixer.Sound(startup_sound_path)
shutdown_sound_effect = pygame.mixer.Sound(shutdown_sound_path)

# Play the startup sound one time
for _ in range(1):
    startup_sound_effect.play()
    time.sleep(startup_sound_effect.get_length())

############################# Camera Setup #############################
class Camera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        self.window_size = (640, 480)
        self.running = self.scroll_enabled = self.overlay_enabled = self.night_vision_enabled = False
        self.text, self.font, self.thread, self.start_time = "", cv2.FONT_HERSHEY_SIMPLEX, None, None
        self.code_lines = [
            "import cv2", "import numpy as np", "cap = cv2.VideoCapture(0)", "while True:",
            "    ret, frame = cap.read()", "    if not ret:", "        break",
            "    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)", "    cv2.imshow('frame', gray)",
            "    if cv2.waitKey(1) & 0xFF == ord('q'):", "        break",
            "cap.release()", "cv2.destroyAllWindows()"
        ]
        self.scroll_index = 0

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

    def run(self):
        self.start_time = time.time()
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
            if self.overlay_enabled:
                frame = self.apply_overlay(frame)
            if self.text and (time.time() - self.start_time) > 5:
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
                self.scroll_enabled = not self.scroll_enabled
            elif key == ord('t'):
                if not self.overlay_enabled:
                    sound_effect.play()
                self.overlay_enabled = not self.overlay_enabled
            elif key == ord('f'):
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, not self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
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
camera.start()

############################# Main Loop #############################
try:
    running = True
    while running:
        distance = distance_measurement()
        print("Distance:", distance)
        camera.text = f"Distance: {distance} inches"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    shutdown_sound_effect.play()
                    time.sleep(shutdown_sound_effect.get_length())
                    time.sleep(1)
                    running = False
                elif event.key == pygame.K_c:
                    if camera.running:
                        camera.stop()
                    else:
                        camera.start()
                elif event.key == pygame.K_p:
                    camera.scroll_enabled = not camera.scroll_enabled
        pygame.display.update()

except KeyboardInterrupt:
    pass

finally:
    if camera.running:
        camera.stop()
    pygame.quit()
