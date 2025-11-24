import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gymtonic/envs')))
from aaron_soccer import SoccerSingleEnv

seed = 42
checkpoint_dir = "iker_checkpoints"
best_model_path = os.path.join(checkpoint_dir, "best_model")
os.makedirs(checkpoint_dir, exist_ok=True)

# Entrenamiento
train = True
load_model = False  # Cambiar a True si quieres continuar entrenamiento
total_timesteps = 500_000

# ---------------------------
# ENVIRONMENT
# ---------------------------
def make_env(render_mode=None):
    env = SoccerSingleEnv(render_mode=render_mode, goal_target='random', perimeter_side=10)
    env = Monitor(env)  
    return env

# Entrenamiento en modo rgb_array para renderizado interno
train_env = make_env(render_mode=None)

# ---------------------------
# CHECKPOINTS Y CALLBACKS
# ---------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,  
    save_path=checkpoint_dir,
    name_prefix='ppo_soccer'
)

# EvalCallback para guardar el mejor modelo
eval_env = make_env(render_mode=None)
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=80, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    best_model_save_path=best_model_path,
    log_path=checkpoint_dir,
    eval_freq=10_000,
    deterministic=True,
    render=False
)

# ---------------------------
# LOAD OR CREATE MODEL
# ---------------------------
if load_model and os.path.exists(os.path.join(checkpoint_dir, "ppo_soccer.zip")):
    model = PPO.load(os.path.join(checkpoint_dir, "ppo_soccer.zip"), env=train_env, seed=seed, verbose=1)
else:
    model = PPO("MlpPolicy", train_env, seed=seed, verbose=1)

# ---------------------------
# TRAIN
# ---------------------------
if train:
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=not load_model,
        progress_bar=True,
        callback=[checkpoint_callback, eval_callback]
    )
    model.save(os.path.join(checkpoint_dir, "ppo_soccer_final"))
train_env.close()

# ---------------------------
# EVALUACIÓN
# ---------------------------
eval_env = make_env(render_mode='human')  
model = PPO.load(os.path.join(best_model_path, "best_model.zip"), env=eval_env, seed=seed, verbose=1)

n_episodes = 10
avg_reward = 0
for ep in range(n_episodes):
    obs, _ = eval_env.reset()
    terminated = truncated = False
    total_reward = 0
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
    print(f"Episode {ep+1} reward: {total_reward}")
    avg_reward += total_reward

print(f"Average reward over {n_episodes} episodes: {avg_reward/n_episodes}")
eval_env.close()