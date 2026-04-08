# Evaluation settings
MIN_STEPS_FOR_CONDITIONAL_WIN = 4
EVAL_EPISODES = 30  # number of greedy episodes for policy evaluation

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer

# Learning settings
BATCH_SIZE = 512
LEARN_RATE = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma
TRAIN_EVERY_N_CLICKS = 10
UPDATE_TARGET_EVERY_N_TRAININGS = 5

# Exploration settings
EPSILON_INIT = 0.95
EPSILON_DECAY = 0.9999998
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
