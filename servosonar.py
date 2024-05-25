import time
import pygamec
from gpiozero import DistanceSensor, Servo
import math

# Set up GPIO for ultrasonic sensor using gpiozero
sensor = DistanceSensor(echo=24, trigger=23)

# Set up Pygame for displaying real-time distance
pygame.init()
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)

# Set up servos
servo1 = Servo(12, min_pulse_width=1.0/1000, max_pulse_width=2.0/1000)
servo2 = Servo(16, min_pulse_width=1.0/1000, max_pulse_width=2.0/1000)

def distance_measurement():
    # Measure distance in inches
    distance = sensor.distance * 100  # distance is in meters, converting to centimeters
    distance_inches = distance / 2.54
    return round(distance_inches, 2)

try:
    continuous_servo = True  # Flag to control continuous servo movement
    running = True  # Flag to control the main loop

    while running:
        # Measure distance
        distance = distance_measurement()
        print("Distance:", distance)

        # Check if distance is under 5 inches
        if distance < 5:
            continuous_servo = False  # Stop continuous servo movement
            servo1.value = 0  # Return servo1 to center position
            servo2.value = 0  # Return servo2 to center position
        else:
            continuous_servo = True  # Restart continuous servo movement

        # Render the real-time distance indicator
        text = font.render(f"Distance: {distance} inches", True, (255, 255, 255))
        screen.fill((0, 0, 0))  # Clear the screen
        screen.blit(text, (10, 10))  # Display the text
        pygame.display.update()  # Update the display

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    servo1.value = 0  # Return servo1 to center position
                    servo2.value = 0  # Return servo2 to center position

        if continuous_servo and running:  # Check if continuous servo movement is enabled
            # Run the servo movements continuously
            for i in range(0, 360):
                if not running:
                    break
                servo1.value = math.sin(math.radians(i))  # Set position for first servo
                servo2.value = math.cos(math.radians(i))  # Set position for second servo using a different function
                time.sleep(0.01)

        time.sleep(1)  # Wait for 1 second before the next measurement

except KeyboardInterrupt:
    pass

finally:
    pygame.quit()
