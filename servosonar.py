import RPi.GPIO as GPIO
import time
import pygame
from gpiozero import Servo
import math
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory

# Set up GPIO for ultrasonic sensor
GPIO.setmode(GPIO.BCM)
TRIG = 23
ECHO = 24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Set up Pygame for displaying real-time distance
pygame.init()
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)

# Set up servos
factory = PiGPIOFactory()
servo1 = Servo(12, min_pulse_width=1.3/1000, max_pulse_width=1.6/1000, pin_factory=factory)
servo2 = Servo(16, min_pulse_width=1.5/1000, max_pulse_width=1.7/1000, pin_factory=factory)

def distance_measurement():
    pulse_start = 0
    pulse_end = 0

    GPIO.output(TRIG, False)
    time.sleep(0.5)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    distance_inches = distance / 2.54

    return distance_inches

try:
    continuous_servo = True  # Flag to control continuous servo movement

    while True:
        # Measure distance
        distance = distance_measurement()
        print("Distance:", distance)

        # Check if distance is under 60 inches
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
                break

        if continuous_servo:  # Check if continuous servo movement is enabled
            # Run the servo movements continuously
            for i in range(0, 360):
                servo1.value = math.sin(math.radians(i))  # Set position for first servo
                servo2.value = math.cos(math.radians(i))  # Set position for second servo using a different function
                sleep(0.01)

        time.sleep(1)  # Wait for 1 second before the next measurement

except KeyboardInterrupt:
    GPIO.cleanup()