#!/usr/bin/env python3
import time
import pygame
from gpiozero import Servo
import math
import os
import threading

# Servo Setup
servos = [Servo(pin, min_pulse_width=1/1000, max_pulse_width=2/1000) for pin in (12, 16, 20)]

# Pygame Setup
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
pygame.init()
pygame.display.set_caption('Servo Control')
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)

sound_paths = {
    pygame.K_F1: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - power.wav',
    pygame.K_F2: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - sequence.wav',
    pygame.K_F4: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - autopilot.wav',
    pygame.K_F6: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - left.wav',
    pygame.K_F7: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - accelerate.wav',
    pygame.K_F8: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - warning.wav',
    pygame.K_F9: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - right.wav',
    pygame.K_q: '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - Disabled.wav',
    'startup': '/home/aounit1/Desktop/AO/AO Sounds/Robot Voice - warning.wav'
}

sounds = {k: pygame.mixer.Sound(v) for k, v in sound_paths.items()}
sounds[pygame.K_F7].set_volume(0.25)
sounds[pygame.K_F6].set_volume(0.15)
sounds[pygame.K_F9].set_volume(0.15)

for _ in range(3):
    sounds['startup'].play()
    time.sleep(1.5)

continuous_servo = False
running = True

def continuous_servo_movement():
    global continuous_servo
    while True:
        if continuous_servo:
            for i in range(360):
                if not continuous_servo:
                    break
                servos[0].value = math.sin(math.radians(i))
                servos[1].value = math.cos(math.radians(i))
                servos[2].value = (math.sin(math.radians(i)) + math.cos(math.radians(i))) / 2
                time.sleep(0.01)
        time.sleep(0.1)

threading.Thread(target=continuous_servo_movement, daemon=True).start()

try:
    while running:
        screen.fill((0, 0, 0))
        screen.blit(font.render("Servo Control", True, (0, 255, 0)), (10, 10))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    servos[0].value = servos[1].value = servos[2].value = 0
                    sounds[event.key].play()
                    time.sleep(2)
                running = False
            elif event.type == pygame.KEYDOWN and event.key in sounds:
                if event.key == pygame.K_F8 and not sounds[pygame.K_F8].get_num_channels():
                    sounds[pygame.K_F8].play(-1)
                elif event.key == pygame.K_F4:
                    continuous_servo = not continuous_servo
                    if continuous_servo:
                        sounds[event.key].play()
                else:
                    sounds[event.key].play()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_F7:
                    sounds[event.key].stop()
                elif event.key == pygame.K_F8:
                    sounds[pygame.K_F8].stop()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_F7]:
            servos[0].value = 1
        elif keys[pygame.K_F8]:
            servos[0].value = -1
        elif keys[pygame.K_F6]:
            servos[1].value = 1
        elif keys[pygame.K_F9]:
            servos[1].value = -1
        elif keys[pygame.K_F1]:
            servos[2].value = -1
        elif keys[pygame.K_F2]:
            servos[2].value = 1
        else:
            if not continuous_servo:
                servos[0].value = servos[1].value = servos[2].value = 0

        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    pygame.quit()
