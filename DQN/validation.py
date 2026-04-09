import os
import numpy as np

VALIDATION_STATES_PATH = 'validation_states/states.npy'

def load_validation_states(nrows, ncols):
    """Load validation states from disk, or return None if unavailable."""
    if os.path.exists(VALIDATION_STATES_PATH):
        raw = np.load(VALIDATION_STATES_PATH)
        return raw.reshape(-1, nrows, ncols, 1)
    return None


def compute_avg_max_q(model, validation_states):
    """Compute average max Q-value over validation states."""
    q_values_batch = model(validation_states).numpy()
    return round(float(np.mean(np.max(q_values_batch, axis=1))), 4)