"""
Visualize states saved by generate_states.py.

Loads a .npy file of state_im arrays, or a .pkl replay buffer saved by
DQN_agent.py, and renders each state using the MinesweeperEnv GUI.
Press any key or click to advance to the next state.

Usage:
    python visualize_states.py [states.npy|replay.pkl] [--width 9] [--height 9] [--n_mines 10]
"""

import argparse
import pickle
import sys
import numpy as np
import pygame

from minesweeper_env import MinesweeperEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('input', nargs='?', default='validation_states/states.npy')
    parser.add_argument('input', nargs='?', default='replay/conv64x4_dense512x2_y0.1_20391b4b.pkl')
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--n_mines', type=int, default=10)
    return parser.parse_args()


def load_states(path):
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            replay_memory = pickle.load(f)
        # Each entry is (current_state, action, reward, new_state, done)
        states = [transition[0] for transition in replay_memory]
        print(f'Loaded {len(states)} states from {path}')
        return states
    else:
        states = np.load(path)
        print(f'Loaded {len(states)} states from {path}  (shape: {states.shape})')
        return states


def main():
    args = parse_args()

    states = load_states(args.input)

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
