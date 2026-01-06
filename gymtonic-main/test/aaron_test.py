import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_latest_run_id
from datetime import datetime

# Path Configuration

# Base directory of the project 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#Directories for saving models, checkpoints, and TensorBoard logs
POLICIES_DIR = os.path.join(BASE_DIR, "policies")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "tensorboard")

# Create directories if they do not exist
os.makedirs(POLICIES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# Base model name, used for files and checkpoints
MODEL_NAME = "ppo_aaron_soccer"
MODEL_PATH = os.path.join(POLICIES_DIR, MODEL_NAME)

# Training Configuration
seed = 42
train = True
load_model = True
total_timesteps = 1_000_000

# Gym Environment Setup
sys.path.append(BASE_DIR)
import gymtonic

def make_env(render_mode=None):
    env = gym.make("gymtonic/aaron_soccer", render_mode=render_mode)
    return Monitor(env)

# Checkpoint Callback Setup
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=CHECKPOINT_DIR,
    name_prefix=MODEL_NAME
)

#  Load Latest Checkpoint Function
def load_latest_checkpoint(env):
    checkpoints = [
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith(".zip")
    ]
    if not checkpoints:
        print(" No hay checkpoints, entrenando desde cero")
        return None

    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f" Cargando checkpoint: {latest_checkpoint}")

    # Load checkpoint and pass tensorboard_log
    model = PPO.load(
        latest_checkpoint,
        env=env,
        seed=seed,
        tensorboard_log=os.path.join(TENSORBOARD_DIR, "from_checkpoint")
    )
    return model



# Training
if train:
    env = make_env()

    model = None
    if load_model:
        model = load_latest_checkpoint(env)

    if model is None:
        # Train a new model from scratch
        experiment_name = datetime.now().strftime("PPO_%Y%m%d_%H%M%S")
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            tensorboard_log=os.path.join(TENSORBOARD_DIR, experiment_name)
        )
        reset_timesteps = True
    else:
        # Continue training from a loaded checkpoint
        reset_timesteps = False

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=1 
    )

    model.save(MODEL_PATH)
    env.close()

# Evaluation
# Create environment with rendering enabled
env = make_env(render_mode="human")

if load_model:
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH, env=env)
    else:
        model = load_latest_checkpoint(env)
else:
    raise RuntimeError("Cannot evaluate without a trained model.")

obs, _ = env.reset()
avg_reward = 0.0
n_eval = 20

for ep in range(n_eval):
    terminated = truncated = False
    ep_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward

    print(f"Episode {ep+1}: reward = {ep_reward:.3f}")
    avg_reward += ep_reward
    obs, _ = env.reset()

print(f"\n Average reward ({n_eval} episodes): {avg_reward / n_eval:.3f}")
env.close()
