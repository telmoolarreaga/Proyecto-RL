# aaron_test.py
import sys
import os
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from gymnasium.envs.registration import register

# Asegúrate de que el paquete gymtonic (o la carpeta envs) esté en PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gymtonic')))

# IMPORTANTE: el nombre de archivo es aaron_soccer.py -> módulo gymtonic.envs.aaron_soccer
# No importar directamente la clase aquí; usaremos gym.make() después de registrar.
# Registrar el entorno (solo si no está ya registrado)

# Esto asegura que 'gymtonic' se pueda importar desde la carpeta raíz Proyecto-RL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ENV_ID = "SoccerSingle-v0"
try:
    gym.spec(ENV_ID)
except Exception:
    register(
        id=ENV_ID,
        entry_point='gymtonic.envs.aaron_soccer:SoccerSingleEnv',
        # allow_early_resets=True, # si quieres permitir resets mientras el episodio no ha terminado
    )

# Seed y paths
seed = 42
random.seed(seed)
np.random.seed(seed)

<<<<<<< HEAD
policies_dir = os.path.join(os.path.dirname(__file__), "../policies")
os.makedirs(policies_dir, exist_ok=True)
model_path = os.path.join(policies_dir, "ppo_soccer_single.zip")
os.makedirs(os.path.join(os.path.dirname(__file__), "../aaron_checkpoints"), exist_ok=True)
=======
if train:
    env = gym.make('SoccerSingle-v0', render_mode="none")
    #check_env(env, warn=True) 
    if load_model and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, seed=seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env=env, seed=seed, verbose=1)
    
    model.learn(total_timesteps=1_000_000, reset_num_timesteps=not load_model, progress_bar=True)
    model.save("policies/ppo_soccer_single")
    env.close()
>>>>>>> c8e4c6c5e2fbfde4cc909e4b8dcd376ccd671c4a

# Training config
train_flag = True
load_model_if_exists = True
total_timesteps = 400_000  # puedes ajustar

# ---------- TRAIN ----------
if train_flag:
    print("Creando entorno para entrenamiento (sin render)...")
    env = gym.make(ENV_ID, render_mode=None)
    env = Monitor(env)

    if load_model_if_exists and os.path.exists(model_path):
        print("Cargando modelo existente:", model_path)
        model = PPO.load(model_path, env=env, seed=seed, verbose=1)
    else:
        print("Creando nuevo modelo PPO")
        model = PPO("MlpPolicy", env=env, seed=seed, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=os.path.join(os.path.dirname(__file__), "../aaron_checkpoints/"), name_prefix='ppo_soccer')
    progress_callback = ProgressBarCallback()

    print("Entrenando...")
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=not (os.path.exists(model_path) and load_model_if_exists),
                callback=[checkpoint_callback, progress_callback])
    print("Guardando modelo en:", model_path)
    model.save(model_path)
    env.close()

# ---------- EVALUACIÓN ----------
print("Evaluando modelo en modo humano (10 episodios)...")
env = gym.make(ENV_ID, render_mode='human')
env = Monitor(env)

model = PPO.load(model_path, env=env, seed=seed, verbose=1)

n_eval = 10
sum_reward = 0.0
for ep in range(n_eval):
    obs, _ = env.reset()
    terminated = truncated = False
    episode_reward = 0.0

    # Opcional: inicializar variables de shaping dentro del env si las usa
    if hasattr(env.unwrapped, "init_evaluation_metrics"):
        env.unwrapped.init_evaluation_metrics()

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

    print(f"Episode {ep+1} reward: {episode_reward:.3f}")
    sum_reward += episode_reward

avg_reward = sum_reward / n_eval
print(f"Average reward over {n_eval} episodes: {avg_reward:.3f}")

env.close()
