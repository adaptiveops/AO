import time
import pygame
import math
import cv2
import threading
import os

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
            elif key == ord('w'):  # Press 'w' to toggle auto/manual white balance
                current_white_balance = self.cap.get(cv2.CAP_PROP_AUTO_WB)
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 - current_white_balance)
                print(f"Auto white balance {'enabled' if current_white_balance == 0 else 'disabled'}")

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
    running = True  # Flag to control the main loop

    while running:
        ############################# Pygame Display Update #############################

        screen.fill((0, 0, 0))  # Clear the screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_c:  # Press 'c' to toggle the camera
                    if camera.running:
                        camera.stop()
                    else:
                        camera.start()
                elif event.key == pygame.K_p:  # Press 'p' to toggle the scrolling code
                    camera.scroll_enabled = not camera.scroll_enabled

        pygame.display.update()  # Update the display

except KeyboardInterrupt:
    pass

finally:
    if camera.running:
        camera.stop()  # Ensure the camera stops
    pygame.quit()
