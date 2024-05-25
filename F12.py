import pygame
import subprocess
import sys

# Initialize Pygame
pygame.init()
pygame.display.set_caption('Key Listener')
screen = pygame.display.set_mode((200, 100))
font = pygame.font.Font(None, 36)

running = True
process = None

while running:
    screen.fill((0, 0, 0))  # Clear the screen
    text = font.render("Press F12 to run script", True, (0, 255, 0))  # Green text
    screen.blit(text, (10, 10))
    pygame.display.update()  # Update the display

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F12:
                if process is None or process.poll() is not None:
                    process = subprocess.Popen([sys.executable, '/home/aounit1/Desktop/AO/ops1.py'])
                else:
                    print("Script is already running")
            elif event.key == pygame.K_q:
                running = False

pygame.quit()

if process is not None:
    process.terminate()
    process.wait()
