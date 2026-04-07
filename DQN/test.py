import argparse
import numpy as np
import pygame
from tqdm import tqdm
from keras.models import load_model
from DQN_agent import DQNAgent
from gui_common import wait_for_click
from minesweeper_env import *

MIN_STEPS_FOR_CONDITIONAL_WIN = 4

def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    parser.add_argument('--model_name', type=str, default='conv64x4_dense512x2_y0.1_minlr0.001_20391b4b',
                        help='name of model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to play')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize gameplay step-by-step (default: run headless evaluation)')

    return parser.parse_args()

params = parse_args()

def visualize(env, agent):
    for episode in tqdm(range(1, params.episodes + 1)):
        env.reset()
        done = False
        while not done:
            current_state = env.state_im
            action, q_values = agent.get_action(current_state, explore=False)
            if q_values is not None:
                env.plot_qvalues_and_next_action(action, q_values)
                wait_for_click()
            new_state, reward, done = env.step(action)
            wait_for_click()


def evaluate(env, agent):
    n_safe_tiles = env.ntiles - env.n_mines
    wins = []
    progress_list = []
    reward_list = []
    conditional_wins = []

    for episode in tqdm(range(1, params.episodes + 1)):
        env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            current_state = env.state_im
            action, q_values = agent.get_action(current_state, explore=False)
            new_state, reward, done = env.step(action)
            episode_reward += reward
            steps += 1

        unrevealed = int(np.sum(env.state_im == -0.125))
        revealed = n_safe_tiles - unrevealed
        progress = revealed / n_safe_tiles

        won = 1 if env.n_wins > 0 else 0
        wins.append(won)
        progress_list.append(progress)
        reward_list.append(episode_reward)
        if steps > MIN_STEPS_FOR_CONDITIONAL_WIN:
            conditional_wins.append(won)

    print(f"\nResults over {params.episodes} episodes:")
    print(f"  Win rate:        {np.mean(wins)*100:.1f}%")
    print(f"  Cond. win rate:  {np.mean(conditional_wins)*100:.1f}%  (episodes with >{MIN_STEPS_FOR_CONDITIONAL_WIN} steps, n={len(conditional_wins)})")
    print(f"  Median progress: {np.median(progress_list)*100:.1f}%")
    print(f"  Median reward:   {np.median(reward_list):.3f}")


def main():
    env = MinesweeperEnv(width=9, height=9, n_mines=10, gui=params.visualize)
    agent = DQNAgent(env, params.model_name)
    agent.load_model_and_replay_buffer(prompt=False)

    if params.visualize:
        visualize(env, agent)
    else:
        evaluate(env, agent)


if __name__ == "__main__":
    main()
