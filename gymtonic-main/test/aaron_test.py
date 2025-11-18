#aaron_test
import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
import gymtonic
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import gymtonic
from gymnasium.envs.registration import register



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gymtonic/envs')))

# Importar tu entorno personalizado
from aaron_socccer import SoccerSingleEnv

# Registrar el entorno con Gym
register(
    id='SoccerSingle-v0',
    entry_point='aaron_socccer:SoccerSingleEnv',  
)

seed = 42
model_path = "policies/ppo_soccer_single.zip"
train = True
load_model = True

if train:
    env = gym.make('SoccerSingle-v0', render_mode="rgb_array")
    #check_env(env, warn=True) 
    if load_model and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, seed=seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env=env, seed=seed, verbose=1)
    
    model.learn(total_timesteps=400_000, reset_num_timesteps=not load_model, progress_bar=True)
    model.save("policies/ppo_soccer_single")
    env.close()

env = gym.make('SoccerSingle-v0', render_mode='human')
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
import numpy as np
import pybullet as p

# ----------------------
# Configuración
# ----------------------
seed = 42
model_path = "policies/ppo_soccer_single.zip"
total_timesteps = 1_000_000  # Ajusta según tu tiempo disponible
os.makedirs("policies", exist_ok=True)
os.makedirs("aaron_checkpoints", exist_ok=True)

# ----------------------
# ENTRENAMIENTO
# ----------------------
train = True

if train:
    print("Creando entorno en modo entrenamiento (no render)...")
    env = gym.make('gymtonic/SoccerSingle-v0', render_mode=None)
    env = Monitor(env)

    if os.path.exists(model_path):
        print("Cargando modelo existente para continuar entrenamiento...")
        model = PPO.load(model_path, env=env, seed=seed, verbose=1)
    else:
        print("Creando nuevo modelo PPO...")
        model = PPO("MlpPolicy", env=env, seed=seed, verbose=1)

    # Checkpoints cada 100_000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path='./aaron_checkpoints/',
        name_prefix='ppo_soccer'
    )

    # Barra de progreso
    progress_callback = ProgressBarCallback()

    print("Entrenando modelo PPO para aprender a meter goles...")
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=not os.path.exists(model_path),
        callback=[checkpoint_callback, progress_callback]
    )
    model.save(model_path)
    env.close()
    print(f"Modelo guardado en {model_path}")

# ----------------------
# EVALUACIÓN
# ----------------------
print("Evaluando modelo PPO en modo humano...")
env = gym.make('SoccerSingle-v0', render_mode='human')
env = Monitor(env)

model = PPO.load(model_path, env=env, seed=seed, verbose=1)

avd_reward = 0
for episode in range(10):
    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0

    # Inicializamos distancia para recompensa por cercanía
    player_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.pybullet_player_id)
    goal_pos = np.array([env.unwrapped.perimeter_side/2, 0]) if env.unwrapped.goal_direction == 'right' else np.array([-env.unwrapped.perimeter_side/2, 0])
    env.unwrapped.previous_distance_to_goal = np.linalg.norm(np.array(player_pos[:2]) - goal_pos)

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # ----------------------
        # Recompensas adicionales
        # ----------------------
        # Recompensa por tocar el balón
        if env.unwrapped.player_touched_ball:
            reward += 0.5

        # Recompensa por acercarse al gol
        player_pos, _ = p.getBasePositionAndOrientation(env.unwrapped.pybullet_player_id)
        distance_to_goal = np.linalg.norm(np.array(player_pos[:2]) - goal_pos)
        reward += 0.01 * (env.unwrapped.previous_distance_to_goal - distance_to_goal)
        env.unwrapped.previous_distance_to_goal = distance_to_goal

        total_reward += reward

    print(f"Episode {episode+1} reward: {total_reward}")
    avd_reward += total_reward

print(f"Average reward: {avd_reward/10}")
env.close()
