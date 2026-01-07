import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime


# Setup and directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

POLICIES_DIR = os.path.join(BASE_DIR, "policies_iker")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_iker")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "tensorboard_iker")

os.makedirs(POLICIES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

MODEL_NAME = "ppo_iker_soccer_v2"
MODEL_PATH = os.path.join(POLICIES_DIR, MODEL_NAME)

# Hyperparameters
seed = 42
# True = training, False = only evaluation
train = False  
# True = load existing model if available    
load_model = True    
total_timesteps = 3_000_000
# Etropy coefficient
ent_coef = 0.005      


# Gym Environment
sys.path.append(BASE_DIR)
import gymtonic

def make_env(render_mode=None):
    env = gym.make("gymtonic/soccer", render_mode=render_mode)
    return Monitor(env)


# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=CHECKPOINT_DIR,
    name_prefix=MODEL_NAME
)

# Fuction to load existing model
def load_existing_model(env):
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Cargando modelo existente: {MODEL_PATH}.zip")
        return PPO.load(MODEL_PATH, env=env, tensorboard_log=TENSORBOARD_DIR)
    else:
        print("No se encontró modelo existente. Entrenando desde cero.")
        return None

# Training
if train:
    env = make_env()

    model = None
    if load_model:
        model = load_existing_model(env)

    if model is None:
        # New model
        experiment_name = datetime.now().strftime("PPO_iker_%Y%m%d_%H%M%S")
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=os.path.join(TENSORBOARD_DIR, experiment_name),
            n_steps = 2048,
        )
        reset_timesteps = True
    else:
        # Continue training existing model
        reset_timesteps = False
        print("Continuando entrenamiento desde modelo existente.")

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=1,
    )

    model.save(MODEL_PATH)
    env.close()

# Evaluation
eval_only = not train
env = make_env(render_mode="human")

if load_model or eval_only:
    model = load_existing_model(env)
    if model is None:
        raise RuntimeError("No se puede evaluar sin un modelo entrenado.")

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

print(f"\nAverage reward ({n_eval} episodes): {avg_reward / n_eval:.3f}")
env.close()
