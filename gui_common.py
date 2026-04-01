import os
import signal
import pygame

def wait_for_click():
    if pygame.get_init():
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return
                if event.type == pygame.QUIT:
                    pygame.quit()
                    os.kill(os.getpid(), signal.SIGINT)
                
