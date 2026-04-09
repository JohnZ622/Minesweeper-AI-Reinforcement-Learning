import os
import numpy as np

VALIDATION_STATES_PATH = 'validation_states/states.npy'

def load_validation_states(nrows, ncols):
    """Load validation states from disk, or return None if unavailable."""
    if os.path.exists(VALIDATION_STATES_PATH):
        raw = np.load(VALIDATION_STATES_PATH)
        return raw.reshape(-1, nrows, ncols, 1)
    return None


def compute_max_q_stats(model, validation_states):
    """Compute average max Q-value over validation states."""
    q_values_batch_maxes = np.max(model(validation_states).numpy(), axis=1)
    return [round(float(np.mean(q_values_batch_maxes)), 4), q_values_batch_maxes[0], q_values_batch_maxes[1]] # use first two states for measurement