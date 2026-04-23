import argparse, signal, queue, time, collections
from tqdm import tqdm
import os

from gui_common import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from DQN_agent import *
from eval_loop import *
from common_constants import *
from training_config import TrainingConfig
from validation import *


def _try_ray_report(metrics: dict) -> None:
    """Forward metrics to Ray Tune if running inside a trial, otherwise no-op."""
    try:
        from ray import train as ray_train
        ray_train.report(metrics)
    except Exception:
        pass


def run_training(
    cfg: TrainingConfig,
    model_name: str,
    *,
    visualize_training: bool = False,
    eval_thread: bool = False,
    loss_heatmap: bool = False,
    log_last_layer_input: bool = True,
    max_clicks: int = 0,
    prompt: bool = True,
    handle_sigint: bool = True,
) -> None:
    rewards = {
        'win': cfg.reward_win,
        'lose': cfg.reward_lose,
        'progress': cfg.reward_progress,
        'guess': cfg.reward_guess,
        'no_progress': cfg.reward_no_progress,
    }
    env = MinesweeperEnv(cfg.width, cfg.height, cfg.n_mines, rewards=rewards, gui=visualize_training)
    agent = DQNAgent(env, model_name, cfg=cfg, log_last_layer_input=log_last_layer_input)

    with agent.tensorboard.writer.as_default():
        hp.hparams({
            'learn_rate': cfg.learn_rate,
            'discount': cfg.discount,
            'epsilon_decay': cfg.epsilon_decay,
            'epsilon_init': cfg.epsilon_init,
            'epsilon_min': cfg.epsilon_min,
            'batch_size': cfg.batch_size,
            'conv_units': cfg.conv_units,
            'dense_units': cfg.dense_units,
            'reward_win': cfg.reward_win,
            'reward_lose': cfg.reward_lose,
            'reward_progress': cfg.reward_progress,
            'reward_guess': cfg.reward_guess,
            'reward_no_progress': cfg.reward_no_progress,
        })
        agent.tensorboard.writer.flush()

    n_clicks = agent.load_model_and_replay_buffer(prompt=prompt)
    for var in agent.model.optimizer.variables:
        var.assign(tf.zeros_like(var)) # Reset optimizer state variables to avoid issues with loaded state
    agent.model.optimizer.learning_rate.assign(cfg.learn_rate) # Ensure the loaded model has the correct learning rate
    print(f"Learning rate: {agent.model.optimizer.learning_rate.numpy()}")
    print(agent.model.summary())

    stop_training = False

    if handle_sigint:
        def save_and_exit(_sig, _frame):
            nonlocal stop_training
            print('\nInterrupted — saving replay buffer and model...')
            stop_training = True
        signal.signal(signal.SIGINT, save_and_exit)

    if eval_thread:
        eval_queue = start_eval_thread(
            model_name, agent.model, cfg.width, cfg.height, cfg.n_mines,
            eval_episodes=cfg.eval_episodes,
            min_steps_for_conditional_win=cfg.min_steps_for_conditional_win,
        )
        inline_eval_worker = None
    else:
        eval_queue = None
        eval_tensorboard = create_eval_tensorboard(model_name)
        inline_eval_worker = EvalWorker(
            agent.model, (cfg.width, cfg.height, cfg.n_mines), eval_tensorboard,
            eval_episodes=cfg.eval_episodes,
            min_steps_for_conditional_win=cfg.min_steps_for_conditional_win,
        )

    validation_states = load_validation_states(env.nrows, env.ncols)

    progress_list = collections.deque(maxlen=AGG_STATS_EVERY)
    wins_list = collections.deque(maxlen=AGG_STATS_EVERY)
    ep_rewards = collections.deque(maxlen=AGG_STATS_EVERY)
    conditional_wins_list = collections.deque(maxlen=AGG_STATS_EVERY)
    last_eval_stats: dict = {}
    last_clicks_log = 0
    last_trains_log = 0
    last_clicks_log_time = time.time()
    last_train_time = time.time()
    last_eval_time = time.time()
    time_between_trains = 0.0
    train_duration = 0.0

    episode = 0
    n_trains = 0
    with tqdm(unit='episode') as pbar:
        while not stop_training and (max_clicks == 0 or n_clicks < max_clicks):
            episode += 1
            pbar.update(1)

            env.reset()
            episode_reward = 0
            episode_steps = 0
            past_n_wins = env.n_wins

            done = False
            while not done:
                current_state = env.state_im
                action, q_values = agent.get_action(current_state, explore=True)

                if q_values is not None and visualize_training:
                    env.plot_qvalues_and_next_action(action, q_values)
                    wait_for_user()

                new_state, reward, done = env.step(action)
                n_clicks += 1

                if visualize_training:
                    wait_for_user()

                episode_reward += reward
                episode_steps += 1

                agent.update_replay_memory((current_state, action, reward, new_state, done))

                if n_clicks % cfg.train_every_n_clicks == 0:
                    now = time.time()
                    time_between_trains = now - last_train_time
                    last_train_time = now

                    train_start = time.time()
                    compute_td_errors = n_trains % cfg.grad_log_every_n_trains == 0
                    td_errors, gradients = agent.train(
                        done,
                        n_clicks=n_clicks,
                        compute_td_errors=compute_td_errors,
                        loss_heatmap=loss_heatmap,
                    )
                    train_duration = time.time() - train_start
                    n_trains = agent.n_trains

                    if n_trains % cfg.grad_log_every_n_trains == 0:
                        with agent.tensorboard.writer.as_default():
                            for var in agent.model.trainable_variables:
                                tf.summary.histogram(f'weights/{var.path}', var, step=n_clicks)
                            agent.tensorboard.writer.flush()

                    if gradients is not None:
                        with agent.tensorboard.writer.as_default():
                            for var, grad in zip(agent.model.trainable_variables, gradients):
                                if grad is not None:
                                    tf.summary.histogram(f'gradients/{var.path}', grad, step=n_clicks)
                            agent.tensorboard.writer.flush()

                    if td_errors is not None:
                        agent.tensorboard.step = n_clicks
                        agent.tensorboard.update_stats(
                            td_error_mean=np.mean(td_errors),
                            td_error_max=np.max(td_errors),
                        )

                    if eval_queue is not None:
                        try:
                            eval_queue.put_nowait((agent.model.get_weights(), n_clicks))
                        except queue.Full:
                            pass  # eval still running, skip this update
            # end of episode

            progress_list.append(env.n_progress) # n of non-guess moves
            ep_rewards.append(episode_reward)

            won = env.n_wins > past_n_wins
            wins_list.append(1 if won else 0)
            if episode_steps > cfg.min_steps_for_conditional_win:
                conditional_wins_list.append(won)

            if len(agent.replay_memory) < cfg.mem_size_min:
                continue

            if inline_eval_worker is not None:
                now = time.time()
                if now - last_eval_time >= cfg.eval_interval_seconds:
                    inline_eval_worker.eval_model.set_weights(agent.model.get_weights())
                    last_eval_stats = inline_eval_worker.run_policy_eval_and_post_stats(n_clicks)
                    last_eval_time = now

            if not episode % AGG_STATS_EVERY:
                med_progress = round(np.median(progress_list), 2)
                win_rate = round(np.sum(wins_list) / AGG_STATS_EVERY, 2)
                med_reward = round(np.median(ep_rewards), 2)
                cond_win_rate = round(np.mean(conditional_wins_list), 2) if conditional_wins_list else 0.0

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
                if validation_states is not None:
                    max_q_stats = compute_max_q_stats(agent.model, validation_states)
                    stats['avg_max_q'] = max_q_stats[0]
                    stats['first_state_max_q' ]  = max_q_stats[1]
                    stats['second_state_max_q' ]  = max_q_stats[2]
                agent.tensorboard.update_stats(**stats)


                print_msg = (
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
                )
                if validation_states is not None:
                    print_msg += (
                        f'avg_max_q: {stats["avg_max_q"]:.4f}, '
                        f'first_state_max_q: {stats["first_state_max_q"]:.4f}, '
                        f'second_state_max_q: {stats["second_state_max_q"]:.4f}, '
                    )
                print_msg += f'Mine hit % in replay: {mine_hit_pct_in_replay:.4f}'
                print(print_msg)

                _try_ray_report({**stats, **last_eval_stats, 'n_clicks': n_clicks})

            if not episode % SAVE_MODEL_EVERY:
                agent.save_model_and_replay_buffer(n_clicks)
        # Final save on exit
        agent.save_model_and_replay_buffer(n_clicks)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--n_mines', type=int, default=10)
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--visualize_training', action='store_true', default=False)
    parser.add_argument('--eval_thread', action='store_true', default=False)
    parser.add_argument('--loss_heatmap', action='store_true', default=False)
    parser.add_argument('--log_last_layer_input', action='store_true', default=True)
    parser.add_argument('--max_clicks', type=int, default=0)
    # Hyperparameter overrides — replaces editing common_constants.py by hand
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE)
    parser.add_argument('--discount', type=float, default=DISCOUNT)
    parser.add_argument('--epsilon_decay', type=float, default=EPSILON_DECAY)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--train_every_n_clicks', type=int, default=TRAIN_EVERY_N_CLICKS)
    parser.add_argument('--conv_units', type=int, default=CONV_UNITS)
    parser.add_argument('--dense_units', type=int, default=DENSE_UNITS)
    return parser.parse_args()


def main():
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))
    if tf.config.list_physical_devices('GPU') == []:
        print("No GPU detected. Training may be slow.")
        exit()

    args = parse_args()
    cfg = TrainingConfig(
        width=args.width,
        height=args.height,
        n_mines=args.n_mines,
        learn_rate=args.learn_rate,
        discount=args.discount,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        train_every_n_clicks=args.train_every_n_clicks,
        conv_units=args.conv_units,
        dense_units=args.dense_units,
    )
    print(cfg)
    run_training(
        cfg, args.model_name,
        visualize_training=args.visualize_training,
        eval_thread=args.eval_thread,
        loss_heatmap=args.loss_heatmap,
        log_last_layer_input=args.log_last_layer_input,
        max_clicks=args.max_clicks,
    )


if __name__ == '__main__':
    main()