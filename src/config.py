from dataclasses import dataclass

# Model
X_DIM = 2
HIDDEN_DIM = 256

# SDE
SCHEDULE_K = 0
BETA_MIN = 0.1
BETA_MAX = 20.0
ADJ_COEF = 0.008

if SCHEDULE_K == 0:
    T = 0.99
else:
    T = 1.0

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_ITERATIONS = 200_000
NUM_DATA = 2000
T_MIN = 0.01

# Sampling
N_SAMPLES = 2000
N_STEPS = 2000

# Model file
MODEL_FILE = "score_model.npz"


@dataclass
class TrainingConfig:
    schedule_type: int
    activation: str
    optimizer: str
    learning_rate: float
    hidden_dim: int
    x_dim: int
    beta_min: float
    beta_max: float
    adj_coef: float
    T: float
