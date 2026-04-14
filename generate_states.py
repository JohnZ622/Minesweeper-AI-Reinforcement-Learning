"""
Generate a fixed set of states for evaluating max Q-value averaged over states.

Runs a number of greedy episodes, sampling a few states per episode,
and saves the collected states to a .npy file.

Usage:
    python generate_states.py --model_name <name> [--episodes 50] [--states_per_episode 3] [--output states.npy]
"""

import argparse
import numpy as np
from keras.models import load_model

from minesweeper_env import MinesweeperEnv
from common_constants import *


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a fixed state set for Q-value evaluation')
    parser.add_argument('--model_name', type=str,
                        default='conv64x4_dense512x2_y0.1_minlr0.001_20391b4b',
                        help='Name of model to load from models/<name>.keras')
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--n_mines', type=int, default=10)
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of greedy episodes to run')
    parser.add_argument('--states_per_episode', type=int, default=1,
                        help='Number of states to sample per episode')
    parser.add_argument('--min_states_per_episode', type=int, default=3,
                        help='Min number of states in the episode to qualify that episode for sampling')
    parser.add_argument('--output', type=str, default='validation_states/states.npy',
                        help='Output file path')
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = f'models/{args.model_name}.keras'
    model = load_model(model_path)
    print(f'Loaded model from {model_path}')

    env = MinesweeperEnv(args.width, args.height, args.n_mines)
    nrows, ncols, ntiles = env.nrows, env.ncols, env.ntiles

    def greedy_action(state):
        board = state.reshape(1, ntiles)
        q_values = model(np.reshape(state, (1, nrows, ncols, 1))).numpy()
        q_values[board != -0.125] = np.min(q_values)
        return np.argmax(q_values)

    all_states = []

    for _ in range(args.episodes):
        env.reset()
        done = False
        episode_states = []

        while not done:
            state = env.state_im.copy()
            episode_states.append(state)
            action = greedy_action(state)
            _, _, done = env.step(action)

        # Randomly sample from the episode trajectory
        if len(episode_states) >= args.min_states_per_episode:
            indices = np.random.choice(len(episode_states), size=args.states_per_episode, replace=False)
            sampled = [episode_states[i] for i in indices]

        all_states.extend(sampled)

    states_array = np.array(all_states)
    np.save(args.output, states_array)
    print(f'Saved {len(states_array)} states to {args.output}  (shape: {states_array.shape})')


if __name__ == '__main__':
    main()
