import queue
import threading
import tensorflow as tf

from minesweeper_env import *
from my_tensorboard2 import *

EVAL_EPISODES = 30  # number of greedy episodes for policy evaluation


def create_eval_tensorboard(model_name):
    return ModifiedTensorBoard(
        log_dir=f'logs/{model_name}_eval', profile_batch=0, update_freq=1)


def run_policy_eval(eval_model, eval_env, eval_tensorboard, step):
    """Run EVAL_EPISODES greedy games. Called from the persistent eval thread."""
    ntiles = eval_env.ntiles
    nrows  = eval_env.nrows
    ncols  = eval_env.ncols

    def greedy_action(state):
        board = state.reshape(1, ntiles)
        q_values = eval_model.predict(
            np.reshape(state, (1, nrows, ncols, 1)), verbose=0)
        q_values[board != -0.125] = np.min(q_values)
        return np.argmax(q_values)

    total_progress, total_wins, total_guesses = 0, 0, 0
    for _ in range(EVAL_EPISODES):
        eval_env.reset()
        past_wins = eval_env.n_wins
        done = False
        while not done:
            _, _, done = eval_env.step(greedy_action(eval_env.state_im))
        total_progress += eval_env.n_progress
        total_guesses += eval_env.n_guesses
        if eval_env.n_wins > past_wins:
            total_wins += 1

    avg_progress = round(total_progress / EVAL_EPISODES, 2)
    eval_winrate = round(total_wins / EVAL_EPISODES, 2)
    eval_guessrate = round(total_guesses / EVAL_EPISODES, 2)

    eval_tensorboard.step = step
    eval_tensorboard.update_stats(
        eval_progress_avg=avg_progress,
        eval_winrate=eval_winrate,
        eval_guessrate=eval_guessrate,
    )


def eval_worker(eval_queue, model_template, eval_env_params, eval_tensorboard):
    """Persistent eval thread: loops waiting for (weights, step) from the main thread."""
    eval_model = tf.keras.models.clone_model(model_template)
    eval_env = MinesweeperEnv(*eval_env_params)
    while True:
        weights, step = eval_queue.get()  # blocks until main thread enqueues work
        eval_model.set_weights(weights)
        run_policy_eval(eval_model, eval_env, eval_tensorboard, step)


def start_eval_thread(model_name, model, width, height, n_mines):
    """Create and start the eval thread. Returns (eval_queue, eval_tensorboard)."""
    eval_tensorboard = create_eval_tensorboard(model_name)
    eval_env_params = (width, height, n_mines)
    eval_queue = queue.Queue(maxsize=1)

    eval_thread = threading.Thread(
        target=eval_worker,
        args=(eval_queue, model, eval_env_params, eval_tensorboard),
        daemon=True)
    eval_thread.start()

    return eval_queue
