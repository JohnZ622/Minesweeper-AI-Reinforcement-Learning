import argparse, signal, queue, time
from tqdm import tqdm
import os

from gui_common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf

from DQN_agent import *
from eval_loop import start_eval_thread
from common_constants import *
from validation import *

print("GPU Available: ", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU') == []:
    print("No GPU detected. Training may be slow.")
    exit()

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--model_name', type=str, default=f'{MODEL_NAME}',
                        help='Name of model')
    parser.add_argument('--visualize_training', action='store_true',
                        help='Visualize the training process', default=False)
    parser.add_argument('--eval_thread', action='store_true',
                        help='Run eval thread during training', default=False)

    return parser.parse_args()

params = parse_args()

AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 10_000 # save model and replay every 10,000 episodes

def main():
    env = MinesweeperEnv(params.width, params.height, params.n_mines, gui=params.visualize_training)
    agent = DQNAgent(env, params.model_name)

    n_clicks = agent.load_model_and_replay_buffer(prompt=True)
    agent.epsilon = 0.01

    stop_training = False

    def save_and_exit(_sig, _frame):
        nonlocal stop_training
        print('\nInterrupted — saving replay buffer and model...')
        stop_training = True

    signal.signal(signal.SIGINT, save_and_exit)

    eval_queue = start_eval_thread(
        params.model_name, agent.model, params.width, params.height, params.n_mines
    ) if params.eval_thread else None

    validation_states = load_validation_states(env.nrows, env.ncols)

    progress_list, wins_list, ep_rewards, conditional_wins_list = [], [], [], []
    last_clicks_log = 0
    last_trains_log = 0
    last_clicks_log_time = time.time()
    last_train_time = time.time()

    episode = 0
    n_trains = 0
    with tqdm(unit='episode') as pbar:
        while not stop_training:
            episode += 1
            pbar.update(1)

            env.reset()
            episode_reward = 0
            past_n_wins = env.n_wins

            done = False
            episode_steps = 0
            while not done:
                current_state = env.state_im

                action, q_values = agent.get_action(current_state, explore=True)

                if (q_values is not None) and params.visualize_training:
                    env.plot_qvalues_and_next_action(action, q_values)
                    wait_for_user()

                new_state, reward, done = env.step(action)

                if params.visualize_training:
                    wait_for_user()

                episode_reward += reward
                episode_steps += 1

                agent.update_replay_memory((current_state, action, reward, new_state, done))
                if n_clicks % TRAIN_EVERY_N_CLICKS == 0:
                    now = time.time()
                    time_between_trains = now - last_train_time
                    last_train_time = now

                    train_start = time.time()
                    compute_td_errors = not episode % AGG_STATS_EVERY
                    td_errors = agent.train(update_target= (n_clicks % TRAIN_EVERY_N_CLICKS) % UPDATE_TARGET_EVERY_N_TRAININGS == 0, compute_td_errors=compute_td_errors) # update target every 5
                    train_duration = time.time() - train_start
                    n_trains += 1

                    if eval_queue is not None:
                        try:
                            eval_queue.put_nowait((agent.model.get_weights(), n_clicks))
                        except queue.Full:
                            pass  # eval still running, skip this update

                n_clicks += 1

            progress_list.append(env.n_progress) # n of non-guess moves
            ep_rewards.append(episode_reward)

            won = env.n_wins > past_n_wins
            wins_list.append(1 if won else 0)
            if episode_steps > MIN_STEPS_FOR_CONDITIONAL_WIN:
                conditional_wins_list.append(won)

            if len(agent.replay_memory) < MEM_SIZE_MIN:
                continue

            if not episode % AGG_STATS_EVERY:
                med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
                win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
                med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)
                recent_cond = conditional_wins_list[-AGG_STATS_EVERY:]
                cond_win_rate = round(np.mean(recent_cond), 2) if recent_cond else 0.0

                now = time.time()
                elapsed = now - last_clicks_log_time
                clicks_per_sec = round((n_clicks - last_clicks_log) / elapsed, 1) if elapsed > 0 else 0
                trains_per_sec = round((n_trains - last_trains_log) / elapsed, 2) if elapsed > 0 else 0
                last_clicks_log = n_clicks
                last_trains_log = n_trains
                last_clicks_log_time = now

                agent.tensorboard.step = n_clicks
                buf = agent.replay_memory
                mine_hit_pct_in_replay = round(sum(1 for t in buf if t[2] == -1 and t[4]) / len(buf), 4) if buf else 0.0
                stats = dict(
                    progress_med = med_progress,
                    winrate = win_rate,
                    cond_winrate = cond_win_rate,
                    reward_med = med_reward,
                    learn_rate = agent.learn_rate,
                    epsilon = agent.epsilon,
                    clicks_per_sec = clicks_per_sec,
                    trains_per_sec = trains_per_sec,
                    time_between_trains=time_between_trains,
                    train_duration=train_duration,
                    mine_hit_pct_in_replay=mine_hit_pct_in_replay)
                if td_errors is not None:
                    stats['td_error_mean'] = np.mean(td_errors)
                    stats['td_error_max'] = np.max(td_errors)
                if validation_states is not None:
                    max_q_stats = compute_max_q_stats(agent.model, validation_states)
                    stats['avg_max_q'] = max_q_stats[0]
                    stats['first_state_max_q' ]  = max_q_stats[1]
                    stats['second_state_max_q' ]  = max_q_stats[2]
                agent.tensorboard.update_stats(**stats)


                print(
                    f'Episode: {episode}, '
                    f'n_clicks: {n_clicks} ({clicks_per_sec}/s), '
                    f'n_trains: {n_trains}, ({trains_per_sec}/s), '
                    f'Median progress: {med_progress}, '
                    f'Median reward: {med_reward}, '
                    f'Win rate: {win_rate}, '
                    f'Conditional win rate: {cond_win_rate}, '
                    f'Time between trains: {time_between_trains:.2f}s, '
                    f'Train duration: {train_duration:.2f}s, '
                    f'Epsilon: {agent.epsilon:.4f}, '
                    f'TD Error Mean: {stats["td_error_mean"]:.4f}, '
                    f'TD Error Max: {stats["td_error_max"]:.4f}, '
                    f'avg_max_q: {stats["avg_max_q"]:.4f}, '
                    f'first_state_max_q: {stats["first_state_max_q"]:.4f}, '
                    f'second_state_max_q: {stats["second_state_max_q"]:.4f}, '
                    f'Mine hit % in replay: {mine_hit_pct_in_replay:.4f}'
                )

            if not episode % SAVE_MODEL_EVERY:
                agent.save_model_and_replay_buffer(n_clicks)
        # Final save on exit
        agent.save_model_and_replay_buffer(n_clicks)

if __name__ == "__main__":
    main()
