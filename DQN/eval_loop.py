import os
import queue
import threading
import tensorflow as tf

from minesweeper_env import *
from my_tensorboard2 import *
from common_constants import *

VALIDATION_STATES_PATH = 'validation_states/states.npy'

def create_eval_tensorboard(model_name):
    return ModifiedTensorBoard(
        log_dir=f'logs/{model_name}_eval', profile_batch=0, update_freq=1)


class EvalWorker:
    def __init__(self, model_template, eval_env_params, eval_tensorboard):
        self.eval_model = tf.keras.models.clone_model(model_template)
        self.eval_env = MinesweeperEnv(*eval_env_params)
        self.eval_tensorboard = eval_tensorboard

        nrows, ncols = self.eval_env.nrows, self.eval_env.ncols
        if os.path.exists(VALIDATION_STATES_PATH):
            raw = np.load(VALIDATION_STATES_PATH)
            self.validation_states = raw.reshape(-1, nrows, ncols, 1)
        else:
            self.validation_states = None

    def run_policy_eval_and_post_stats(self, step):
        """Run EVAL_EPISODES greedy games and post stats to tensorboard."""
        ntiles = self.eval_env.ntiles
        nrows  = self.eval_env.nrows
        ncols  = self.eval_env.ncols

        def greedy_action(state):
            board = state.reshape(1, ntiles)
            q_values = self.eval_model(np.reshape(state, (1, nrows, ncols, 1))).numpy()
            q_values[board != -0.125] = np.min(q_values)
            return np.argmax(q_values)

        total_progress, total_wins, total_guesses = 0, 0, 0
        conditional_wins = []
        for _ in range(EVAL_EPISODES):
            self.eval_env.reset()
            past_wins = self.eval_env.n_wins
            done = False
            steps = 0
            while not done:
                _, _, done = self.eval_env.step(greedy_action(self.eval_env.state_im))
                steps += 1
            won = self.eval_env.n_wins > past_wins
            total_progress += self.eval_env.n_progress
            total_guesses += self.eval_env.n_guesses
            if won:
                total_wins += 1
            if steps > MIN_STEPS_FOR_CONDITIONAL_WIN:
                conditional_wins.append(won)

        avg_progress = round(total_progress / EVAL_EPISODES, 2)
        eval_winrate = round(total_wins / EVAL_EPISODES, 2)
        eval_guessrate = round(total_guesses / EVAL_EPISODES, 2)
        eval_cond_winrate = round(np.mean(conditional_wins), 2) if conditional_wins else 0.0

        stats = dict(
            eval_progress_avg=avg_progress,
            eval_winrate=eval_winrate,
            eval_guessrate=eval_guessrate,
            eval_cond_winrate=eval_cond_winrate,
        )

        if self.validation_states is not None:
            q_values_batch = self.eval_model(self.validation_states).numpy()
            avg_max_q = round(float(np.mean(np.max(q_values_batch, axis=1))), 4)
            stats['eval_avg_max_q'] = avg_max_q

        self.eval_tensorboard.step = step
        self.eval_tensorboard.update_stats(**stats)


def eval_worker(eval_queue, model_template, eval_env_params, eval_tensorboard):
    """Persistent eval thread: loops waiting for (weights, step) from the main thread."""
    worker = EvalWorker(model_template, eval_env_params, eval_tensorboard)
    while True:
        weights, step = eval_queue.get()  # blocks until main thread enqueues work
        worker.eval_model.set_weights(weights)
        worker.run_policy_eval_and_post_stats(step)


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
