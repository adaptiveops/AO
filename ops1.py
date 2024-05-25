import time
import pygame
from gpiozero import DistanceSensor, Servo
import math
import cv2
import threading
import os

############################# Distance Sensor and Servo Setup #############################

# Set up GPIO for ultrasonic sensor using gpiozero
sensor = DistanceSensor(echo=24, trigger=23)

# Set up servos
# Adjust the pulse width range to match NSDRC RS400 requirements
servo1 = Servo(12, min_pulse_width=1.0/1000, max_pulse_width=2.0/1000)
servo2 = Servo(16, min_pulse_width=1.0/1000, max_pulse_width=2.0/1000)
servo3 = Servo(20, min_pulse_width=1.0/1000, max_pulse_width=2.0/1000)

def distance_measurement():
    # Measure distance in inches
    distance = sensor.distance * 100  # distance is in meters, converting to centimeters
    distance_inches = distance / 2.54
    return round(distance_inches, 2)

############################# Pygame Setup #############################

# Set up Pygame for capturing key events
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
pygame.init()
pygame.display.set_caption('Distance Measurement')

# Create a Pygame window of a reasonable size for displaying distance
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)

# Load sound effect
sound_path = '/home/aounit1/Desktop/AO/sound-effect-night-vision-2_130bpm_C#.wav'
pygame.mixer.init()
sound_effect = pygame.mixer.Sound(sound_path)

############################# Camera Setup #############################

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
        self.overlay_enabled = False  # Start with the overlay disabled
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
        cv2.line(overlay, (0, height // 2), (width // 4, height // 2), (0, 0, 255), 1)  # Red horizontal line
        cv2.line(overlay, (3 * width // 4, height // 2), (width, height // 2), (0, 0, 255), 1)  # Red horizontal line
        cv2.line(overlay, (width // 2, 0), (width // 2, height // 4), (0, 0, 255), 1)  # Red vertical line
        cv2.line(overlay, (width // 2, 3 * height // 4), (width // 2, height), (0, 0, 255), 1)  # Red vertical line

        # Draw centimeter marks
        cm_interval = 10  # Pixels per centimeter

        # Horizontal line centimeter marks
        for i in range(0, width // 4, cm_interval):
            if i != 0:
                # Centimeter marks
                cv2.line(overlay, (width // 2 + i, height // 2 - 3), (width // 2 + i, height // 2 + 3), (0, 255, 0), 1)
                cv2.line(overlay, (width // 2 - i, height // 2 - 3), (width // 2 - i, height // 2 + 3), (0, 255, 0), 1)

        # Vertical line centimeter marks
        for i in range(0, height // 4, cm_interval):
            if i != 0:
                # Centimeter marks
                cv2.line(overlay, (width // 2 - 3, height // 2 + i), (width // 2 + 3, height // 2 + i), (0, 255, 0), 1)
                cv2.line(overlay, (width // 2 - 3, height // 2 - i), (width // 2 + 3, height // 2 - i), (0, 255, 0), 1)

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
                # Switch servos to hardware driven
                servo1.source = None
                servo2.source = None
                if key == 0x7A:  # F11 key
                    if servo3.source is None:
                        servo3.source = 0  # Turn on servo
                    else:
                        servo1.source = None
                        servo2.source = None
                        if key == 0x7A:  # F11 key
                            servo3.source = None  # Turn off servo
                time.sleep(0.1)  # Add a delay to debounce the input signal
            elif key == ord('s'):  # Press 's' to make the window smaller
                self.window_width = max(20, self.window_width - 20)
                self.window_height = max(20, self.window_height - 20)
            elif key == ord('p'):  # Press 'p' to toggle the scrolling code
                self.scroll_enabled = not self.scroll_enabled
            elif key == ord('t'):  # Press 't' to toggle the overlay
                if not self.overlay_enabled:
                    sound_effect.play()  # Play the sound effect only when turning on the overlay
                self.overlay_enabled = not self.overlay_enabled
                print(f"Overlay enabled: {self.overlay_enabled}")
            elif key == ord('f'):  # Press 'f' to toggle auto/manual focus
                current_focus = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 - current_focus)
                print(f"Auto focus {'enabled' if current_focus == 0 else 'disabled'}")

            # Increase maximum pulse widths for servos to smooth out jitter
            elif key == ord('j'):
                servo1.maximum_pulse_width += 10
                servo2.maximum_pulse_width += 10
                servo3.maximum_pulse_width += 10
                print("Increased maximum pulse widths for servos")
            elif key == ord('w'):  # Press 'w' to toggle auto/manual white balance
                current_white_balance = self.cap.get(cv2.CAP_PROP_AUTO_WB)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 - current_white_balance)
                print(f"Auto white balance {'enabled' if current_white_balance == 0 else 'disabled'}")

            # Handle F keys for servo control
            if key == 0x70:  # F1 key
                servo3.value = -1  # Power position
            elif key == 0x71:  # F2 key
                servo3.value = 1  # Mode position
            elif key == 0x75:  # F6 key
                servo2.value = 1  # Right position (maximum)
            elif key == 0x76:  # F7 key
                servo1.value = 1  # Backward position (maximum)
            elif key == 0x77:  # F8 key
                servo1.value = -1  # Forward position (minimum)
            elif key == 0x78:  # F9 key
                servo2.value = -1  # Left position (minimum)
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
camera.start()

############################# Main Loop #############################

try:
    continuous_servo = False  # Start with continuous servo movement off
    running = True  # Flag to control the main loop

    while running:
        ############################# Distance Measurement #############################

        # Measure distance
        distance = distance_measurement()
        print("Distance:", distance)

        # Update camera text
        camera.text = f"Distance: {distance} inches"

        # Check if distance is under 5 inches
        if (distance < 5):
            servo1.value = 0  # Return servo1 to center position (neutral)
            servo2.value = 0  # Return servo2 to center position (neutral)
            servo3.value = 0  # Return servo3 to center position (neutral)
            disable_keys = True
        else:
            disable_keys = False

        ############################# Pygame Display Update #############################

        # Render the real-time distance indicator
        screen.fill((0, 0, 0))  # Clear the screen
        text = font.render(f"Distance: {distance} inches", True, (0, 255, 0))  # Green text
        screen.blit(text, (10, 10))
        pygame.display.update()  # Update the display

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    servo1.value = 0  # Return servo1 to center position (neutral)
                    servo2.value = 0  # Return servo2 to center position (neutral)
                    servo3.value = 0  # Return servo3 to center position (neutral)
                elif event.key == pygame.K_F1:
                    servo3.value = -1  # Power position
                elif event.key == pygame.K_F2:
                    servo3.value = 1  # Mode position
                elif event.key == pygame.K_F6:
                    servo2.value = 1  # Right position (maximum)
                elif event.key == pygame.K_F7:
                    servo1.value = 1  # Backward position (maximum)
                elif event.key == pygame.K_F8:
                    servo1.value = -1  # Forward position (minimum)
                elif event.key == pygame.K_F9:
                    servo2.value = -1  # Left position (minimum)
                elif event.key == pygame.K_F4:
                    continuous_servo = not continuous_servo  # Toggle continuous servo movement
                    if not continuous_servo:
                        servo1.value = 0  # Return servo1 to center position (neutral)
                        servo2.value = 0  # Return servo2 to center position (neutral)
                        servo3.value = 0  # Return servo3 to center position (neutral)
                elif event.key == pygame.K_c:  # Press 'c' to toggle the camera
                    if camera.running:
                        camera.stop()
                    else:
                        camera.start()
                elif event.key == pygame.K_p:  # Press 'p' to toggle the scrolling code
                    camera.scroll_enabled = not camera.scroll_enabled

        keys = pygame.key.get_pressed()
        if not disable_keys:
            if keys[pygame.K_F1]:
                servo3.value = -1  # Power position
            elif keys[pygame.K_F2]:
                servo3.value = 1  # Mode position
            elif keys[pygame.K_F6]:
                servo2.value = 1  # Right position (maximum)
            elif keys[pygame.K_F7]:
                servo1.value = 1  # Backward position (maximum)
            elif keys[pygame.K_F8]:
                servo1.value = -1  # Forward position (minimum)
            elif keys[pygame.K_F9]:
                servo2.value = -1  # Left position (minimum)
            else:
                if not continuous_servo:
                    servo1.value = 0  # Center position
                    servo2.value = 0  # Center position
                    servo3.value = 0  # Center position
        else:
            servo1.value = 0  # Center position
            servo2.value = 0  # Center position
            servo3.value = 0  # Center position

        if continuous_servo and not disable_keys:
            # Run the servo movements continuously
            for i in range(0, 360):
                if not continuous_servo or disable_keys:
                    break
                servo1.value = math.sin(math.radians(i))  # Set position for servo1 (forward/backward)
                servo2.value = math.cos(math.radians(i))  # Set position for servo2 (left/right)
                servo3.value = (math.sin(math.radians(i)) + math.cos(math.radians(i))) / 2  # Set position for servo3 (power/mode)
                time.sleep(0.01)  # Add a small delay between each servo movement
        time.sleep(0.1)  # Shortened sleep time for quicker response

except KeyboardInterrupt:
    pass

finally:
    if camera.running:
        camera.stop()  # Ensure the camera stops
    pygame.quit()
