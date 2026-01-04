import sys
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Añadir gymtonic-main al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Registrar entorno si aún no está registrado
import gymtonic

# ---------- CALLBACK PARA CHECKPOINT ----------
class SaveOnStepCallback(BaseCallback):
    """Guarda el modelo cada `save_freq` timesteps."""
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Checkpoint guardado en {save_file}")
        return True

# ---------------- CONFIG ----------------
seed = 42
train = False
load_model = True
checkpoint_dir = "checkpoints"
checkpoint_callback = SaveOnStepCallback(save_freq=50_000, save_path=checkpoint_dir)
model_path = "policies/ppo_aaron_soccer"

# ---------- ENTORNO DE ENTRENAMIENTO ----------
if train:
    env = gym.make('gymtonic/aaron_soccer', render_mode='rgb_array')
    env = Monitor(env)

    if load_model and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, seed=seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env=env, seed=seed, verbose=1)

    model.learn(
        total_timesteps=1_000_000,
        reset_num_timesteps=not (load_model and os.path.exists(model_path + ".zip")),
        progress_bar=True,
        callback=checkpoint_callback
    )

    model.save(model_path)
    env.close()

# ---------- EVALUACIÓN ----------
env = gym.make('gymtonic/aaron_soccer', render_mode='human')  # ⚡ render para ver obstáculos y portero
env = Monitor(env)

model = PPO.load(model_path, env=env, seed=seed, verbose=1)

# ⚡ Reset explícito para que se creen obstáculos y portero
obs, info = env.reset()

avg_reward = 0.0
n_eval = 20  # puedes subir a 100 para métricas más precisas

for ep in range(n_eval):
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Episode {ep+1} reward: {total_reward:.3f}")
    avg_reward += total_reward

    # Reset entre episodios
    obs, info = env.reset()

print(f"Average reward over {n_eval} episodes: {avg_reward/n_eval:.3f}")

env.close()
