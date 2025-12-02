import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.env_checker import check_env

# Añadir gymtonic-main al PYTHONPATH para asegurar que aaron_soccer.py se importe
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Registrar tu entorno si aún no está registrado
import gymnasium

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import gymtonic

seed = 42

train = True
load_model = False

if train:
    env = gym.make('gymtonic/aaron_soccer', render_mode="rgb_array")
    #check_env(env, warn=True) 
    if load_model:
        model = PPO.load("policies/ppo_soccer_single", env, seed=seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env, seed=seed, verbose=1)
    
    model.learn(total_timesteps=400_000, reset_num_timesteps=not load_model, progress_bar=True)
    model.save("policies/ppo_aaron_soccer")
    env.close()

env = gym.make('gymtonic/SoccerSingle-v0', render_mode='none')
env = Monitor(env)
model = PPO.load("policies/ppo_soccer_single", env, seed=seed, verbose=1)

avd_reward = 0

for _ in range(100):
    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0

    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    avd_reward += total_reward
print(f"Average reward: {avd_reward/100}")


# Pybullet rendering not working with evaluate_policy
#mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, render=True)
#print(f"Mean_reward:{mean_reward:.2f}")

env.close()
seed = 42
train = True
load_model = True
model_path = "policies/ppo_aaron_soccer"

# ---------- TRAIN ----------
if train:
    env = gym.make("gymtonic/SoccerSingleAaron-v0", render_mode=None)
    env = Monitor(env)

    # check_env(env, warn=True)  # opcional: revisar entorno

    if load_model and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, seed=seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env=env, seed=seed, verbose=1)

    model.learn(total_timesteps=400_000, reset_num_timesteps=not (load_model and os.path.exists(model_path + ".zip")), progress_bar=True)
    model.save(model_path)
    env.close()

# ---------- EVALUACIÓN ----------
env = gym.make("gymtonic/SoccerSingleAaron-v0", render_mode=None)
env = Monitor(env)
model = PPO.load(model_path, env=env, seed=seed, verbose=1)

avg_reward = 0.0
n_eval = 100

for _ in range(n_eval):
    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    print(f"Episode reward: {total_reward:.3f}")
    avg_reward += total_reward

print(f"Average reward over {n_eval} episodes: {avg_reward/n_eval:.3f}")

env.close()
