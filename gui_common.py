import os
import signal
import pygame

def wait_for_user():
    if pygame.get_init():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    return True
                if event.type == pygame.QUIT:
                    pygame.quit()
                    os.kill(os.getpid(), signal.SIGINT)
                
