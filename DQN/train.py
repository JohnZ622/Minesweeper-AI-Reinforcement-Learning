import argparse, signal, queue, time
from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf

from DQN_agent import *
from eval_loop import start_eval_thread

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

    n_clicks = agent.load_model_and_replay_buffer()

    stop_training = False

    def save_and_exit(_sig, _frame):
        nonlocal stop_training
        print('\nInterrupted — saving replay buffer and model...')
        stop_training = True

    signal.signal(signal.SIGINT, save_and_exit)

    eval_queue = start_eval_thread(
        params.model_name, agent.model, params.width, params.height, params.n_mines)

    progress_list, wins_list, ep_rewards = [], [], []
    last_clicks_log = 0
    last_clicks_log_time = time.time()
    last_train_time = time.time()

    episode = 0
    with tqdm(unit='episode') as pbar:
        while not stop_training:
            episode += 1
            pbar.update(1)
            agent.tensorboard.step = n_clicks

            env.reset()
            episode_reward = 0
            past_n_wins = env.n_wins

            done = False
            while not done:
                current_state = env.state_im

                action = agent.get_action(current_state)

                new_state, reward, done = env.step(action)

                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                if n_clicks % 500 == 0:
                    now = time.time()
                    time_between_trains = now - last_train_time
                    last_train_time = now

                    train_start = time.time()
                    agent.train(done)
                    train_duration = time.time() - train_start

                    agent.tensorboard.update_stats(
                        time_between_trains=time_between_trains,
                        train_duration=train_duration,
                    )
                    try:
                        eval_queue.put_nowait((agent.model.get_weights(), n_clicks))
                    except queue.Full:
                        pass  # eval still running, skip this update

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

                now = time.time()
                elapsed = now - last_clicks_log_time
                clicks_per_sec = round((n_clicks - last_clicks_log) / elapsed, 1) if elapsed > 0 else 0
                last_clicks_log = n_clicks
                last_clicks_log_time = now

                agent.tensorboard.update_stats(
                    progress_med = med_progress,
                    winrate = win_rate,
                    reward_med = med_reward,
                    learn_rate = agent.learn_rate,
                    epsilon = agent.epsilon,
                    clicks_per_sec = clicks_per_sec)

                print(f'Episode: {episode}, n_clicks: {n_clicks} ({clicks_per_sec}/s), Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

            if not episode % SAVE_MODEL_EVERY:
                agent.save_model_and_replay_buffer(n_clicks)
        # Final save on exit
        agent.save_model_and_replay_buffer(n_clicks)

if __name__ == "__main__":
    main()
