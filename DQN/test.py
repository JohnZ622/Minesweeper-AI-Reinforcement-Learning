import argparse
import pygame
from tqdm import tqdm
from keras.models import load_model
from DQN_agent import DQNAgent
from minesweeper_env import *

def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    parser.add_argument('--model_name', type=str, default='conv64x4_dense512x2_y0.1_minlr0.001_20391b4b',
                        help='name of model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to play')

    return parser.parse_args()

params = parse_args()

def wait_for_click():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

def main():
    env = MinesweeperEnv(width=9, height=9, n_mines=10, gui=True)
    agent = DQNAgent(env, params.model_name)
    agent.load_model_and_replay_buffer(prompt=False)

    for episode in tqdm(range(1, params.episodes+1)):
        env.reset()

        done = False
        while not done:
            current_state = env.state_im
            action, q_values = agent.get_action(current_state, explore=False)
            if (q_values is not None):
                env.plot_qvalues(q_values)

                wait_for_click()

            new_state, reward, done = env.step(action)
            wait_for_click()


if __name__ == "__main__":
    main()
