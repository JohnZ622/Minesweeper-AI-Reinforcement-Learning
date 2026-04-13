import os, sys, pickle

ROOT = os.getcwd()
sys.path.insert(1, f'{os.path.dirname(ROOT)}')

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

from collections import deque
from minesweeper_env import *
# use my_tensorboard2.py if using tensorflow v2+, use my_tensorboard.py otherwise
from my_tensorboard2 import *
from DQN import *

from common_constants import *

# Default model name
MODEL_NAME = f'conv{CONV_UNITS}x4_dense{DENSE_UNITS}x2_y{DISCOUNT}_minlr{LEARN_MIN}_20391b4b'

class DQNAgent(object):
    def __init__(self, env, model_name, conv_units=CONV_UNITS, dense_units=DENSE_UNITS):
        self.env = env
        self.model_name = model_name

        # Deep Q-learning Parameters
        self.learn_rate = LEARN_RATE
        self.epsilon = EPSILON_INIT
        self.model = create_dqn(
            self.learn_rate, self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)

        # target model - this is what we predict against every step
        self.target_model = create_dqn(
            self.learn_rate, self.env.state_im.shape, self.env.ntiles, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.n_trains = 0

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f'logs/{model_name}', profile_batch=0, update_freq=50)

    def get_action(self, state, explore=False):
        board = state.reshape(1, self.env.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon and explore is True: # random move (explore)
            move = np.random.choice(unsolved)
            q_values = None
        else:
            q_values = self.model(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1))).numpy()
            q_values[board!=-0.125] = np.min(q_values) # set already clicked tiles to min value
            move = np.argmax(q_values)

        return move, q_values

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, update_target, compute_td_errors=False):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return None

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_qs_list = self.model(current_states, batch_size=len(current_states), verbose = 0).numpy()

        new_current_states = np.array([transition[3] for transition in batch])
        future_qs_list = self.target_model(new_current_states, batch_size=len(new_current_states), verbose = 0).numpy()

        X,y = [], []
        td_errors = []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            old_q = current_qs[action]
            current_qs[action] = new_q

            # Compute TD-error: |new_q - old_q|
            td_error = abs(new_q - old_q)
            td_errors.append(td_error)

            X.append(current_state)
            y.append(current_qs)

        X_arr = np.array(X)
        y_arr = np.array(y)

        if self.n_trains % GRAD_LOG_EVERY_N_TRAINS == 0:
            with tf.GradientTape() as tape:
                preds = self.model(X_arr, training=True)
                loss = self.model.compiled_loss(y_arr, preds)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.model.compiled_metrics.update_state(y_arr, preds)
        else:
            gradients = None
            self.model.fit(X_arr, y_arr, batch_size=BATCH_SIZE,
                           shuffle=False, verbose=0, callbacks=[self.tensorboard]\
                           if done else None)

        self.n_trains += 1

        # updating to determine if we want to update target_model yet
        if update_target:
            self.target_model.set_weights(self.model.get_weights())

        # decay learn_rate
        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)

        if compute_td_errors:
            return np.array(td_errors), gradients
        return None, gradients

    def load_model_and_replay_buffer(self, prompt: bool = True):
        n_clicks = 0
        model_path = f'models/{self.model_name}.keras'
        replay_path = f'replay/{self.model_name}.pkl'
        step_path = f'replay/{self.model_name}.step'

        if os.path.exists(model_path):
            if prompt:
                response = input(f"Model file found: '{model_path}'. Load it? [y=load / n=erase]: ").strip().lower()
            else:
                response = 'y'
            if response == 'y':
                from keras.models import load_model
                self.model = load_model(model_path)
                self.target_model = load_model(model_path)
                print(f'Loaded model from {model_path}')
            else:
                print('Model will be overwritten.')

        if os.path.exists(replay_path):
            if prompt:
                response = input(f"Replay buffer found: '{replay_path}'. Load it? [y=load / n=erase]: ").strip().lower()
            else:
                response = 'y'
            if response == 'y':
                with open(replay_path, 'rb') as f:
                    self.replay_memory = pickle.load(f)
                print(f'Loaded replay buffer from {replay_path} ({len(self.replay_memory)} entries)')
            else:
                print('Replay buffer will be overwritten.')

        if os.path.exists(step_path):
            with open(step_path, 'r') as f:
                n_clicks = int(f.read().strip())
            print(f'Loaded step counter: {n_clicks}')

        return n_clicks

    def save_model_and_replay_buffer(self, n_clicks):
        os.makedirs('replay', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        with open(f'replay/{self.model_name}.pkl', 'wb') as output:
            pickle.dump(self.replay_memory, output)
        self.model.save(f'models/{self.model_name}.keras')
        with open(f'replay/{self.model_name}.step', 'w') as f:
            f.write(str(n_clicks))

if __name__ == "__main__":
    DQNAgent(MinesweeperEnv(9,9,10))
