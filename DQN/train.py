import argparse, pickle, signal, sys
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf

from keras.models import load_model
from DQN_agent import *

print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=100_000,
                        help='Number of episodes to train on')
    parser.add_argument('--model_name', type=str, default=f'{MODEL_NAME}',
                        help='Name of model')

    return parser.parse_args()

params = parse_args()

AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 10_000 # save model and replay every 10,000 episodes

def main():
    env = MinesweeperEnv(params.width, params.height, params.n_mines)
    agent = DQNAgent(env, params.model_name)

    model_path = f'models/{params.model_name}.keras'
    replay_path = f'replay/{params.model_name}.pkl'

    if os.path.exists(model_path):
        response = input(f"Model file found: '{model_path}'. Load it? [y=load / n=erase]: ").strip().lower()
        if response == 'y':
            agent.model = load_model(model_path)
            agent.target_model = load_model(model_path)
            print(f'Loaded model from {model_path}')
        else:
            print('Model will be overwritten.')

    if os.path.exists(replay_path):
        response = input(f"Replay buffer found: '{replay_path}'. Load it? [y=load / n=erase]: ").strip().lower()
        if response == 'y':
            with open(replay_path, 'rb') as f:
                agent.replay_memory = pickle.load(f)
            print(f'Loaded replay buffer from {replay_path} ({len(agent.replay_memory)} entries)')
        else:
            print('Replay buffer will be overwritten.')

    def save_and_exit(_sig, _frame):
        print('\nInterrupted — saving replay buffer and model...')
        os.makedirs('replay', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        with open(f'replay/{params.model_name}.pkl', 'wb') as output:
            pickle.dump(agent.replay_memory, output)
        agent.model.save(f'models/{params.model_name}.keras')
        print('Saved.')
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)

    progress_list, wins_list, ep_rewards = [], [], []
    n_clicks = 0

    episode = 0
    with tqdm(unit='episode') as pbar:
        while True:
            episode += 1
            pbar.update(1)
            agent.tensorboard.step = episode

            env.reset()
            episode_reward = 0
            past_n_wins = env.n_wins

            done = False
            new_transitions_count = 0
            while not done:
                current_state = env.state_im

                action = agent.get_action(current_state)

                new_state, reward, done = env.step(action)

                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                if new_transitions_count == 5000:
                    agent.train(done)
                    new_transitions_count = 0

                n_clicks += 1

            progress_list.append(env.n_progress) # n of non-guess moves
            ep_rewards.append(episode_reward)

            if env.n_wins > past_n_wins:
                wins_list.append(1)
            else:
                wins_list.append(0)

            if len(agent.replay_memory) < MEM_SIZE_MIN:
                continue

            if not episode % AGG_STATS_EVERY:
                med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
                win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
                med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

                agent.tensorboard.update_stats(
                    progress_med = med_progress,
                    winrate = win_rate,
                    reward_med = med_reward,
                    learn_rate = agent.learn_rate,
                    epsilon = agent.epsilon)

                print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

            if not episode % SAVE_MODEL_EVERY:
                os.makedirs('replay', exist_ok=True)
                os.makedirs('models', exist_ok=True)
                with open(f'replay/{params.model_name}.pkl', 'wb') as output:
                    pickle.dump(agent.replay_memory, output)

                agent.model.save(f'models/{params.model_name}.keras')

if __name__ == "__main__":
    main()
