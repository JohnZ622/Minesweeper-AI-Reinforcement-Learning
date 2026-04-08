"""
Visualize states saved by generate_states.py.

Loads a .npy file of state_im arrays and renders each one using the
MinesweeperEnv GUI. Press any key or click to advance to the next state.

Usage:
    python visualize_states.py [states.npy] [--width 9] [--height 9] [--n_mines 10]
"""

import argparse
import sys
import numpy as np
import pygame

from minesweeper_env import MinesweeperEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default='validation_states/states.npy')
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--n_mines', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    states = np.load(args.input)
    print(f'Loaded {len(states)} states from {args.input}  (shape: {states.shape})')

    env = MinesweeperEnv(args.width, args.height, args.n_mines, gui=True)

    for i, state_im in enumerate(states):
        env.set_state_im(state_im)
        env._render()
        pygame.display.set_caption(f'Minesweeper — state {i+1}/{len(states)}')

        # Wait for keypress or mouse click to advance; Q or window-close to quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    waiting = False

    pygame.quit()


if __name__ == '__main__':
    main()
