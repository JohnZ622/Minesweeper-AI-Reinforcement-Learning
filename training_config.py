from dataclasses import dataclass, fields
from common_constants import *

@dataclass
class TrainingConfig:
    """All hyperparameters for a single training run. Defaults mirror common_constants.py."""
    # Board
    width: int = 9
    height: int = 9
    n_mines: int = 10
    # Memory
    mem_size: int = MEM_SIZE
    mem_size_min: int = MEM_SIZE_MIN
    # Learning
    batch_size: int = BATCH_SIZE
    learn_rate: float = LEARN_RATE
    discount: float = DISCOUNT
    train_every_n_clicks: int = TRAIN_EVERY_N_CLICKS
    update_target_every_n_episodes: int = UPDATE_TARGET_EVERY_N_EPISODES
    # Exploration
    epsilon_init: float = EPSILON_INIT
    epsilon_decay: float = EPSILON_DECAY
    epsilon_min: float = EPSILON_MIN
    # Architecture
    conv_units: int = CONV_UNITS
    dense_units: int = DENSE_UNITS
    # Rewards
    reward_win: float = REWARD_WIN
    reward_lose: float = REWARD_LOSE
    reward_progress: float = REWARD_PROGRESS
    reward_guess: float = REWARD_GUESS
    reward_no_progress: float = REWARD_NO_PROGRESS
    # Logging
    grad_log_every_n_trains: int = GRAD_LOG_EVERY_N_TRAINS
    eval_interval_seconds: float = EVAL_INTERVAL_SECONDS
    # Evaluation
    eval_episodes: int = EVAL_EPISODES
    min_steps_for_conditional_win: int = MIN_STEPS_FOR_CONDITIONAL_WIN

    def __str__(self) -> str:
        lines = ['TrainingConfig:']
        for f in fields(self):
            lines.append(f'  {f.name}: {getattr(self, f.name)}')
        return '\n'.join(lines)