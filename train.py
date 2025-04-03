import gymnasium as gym

from stable_baselines3 import PPO
from custom_environment import ImageExplorationEnv
from PIL import Image
import numpy as np
from time import sleep


image_path = "N17E073.jpg"
img = Image.open(image_path).convert("L").resize((256, 256))
map_array = np.array(img, dtype=np.uint8)
env = ImageExplorationEnv(map_array, max_steps=700, render_mode="human")


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_scalarfield_tensorboard/")
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs, _ = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = vec_env.step(action)

    vec_env.render()
    # VecEnv resets automatically
    if done:
      obs = env.reset()

env.close()

model.save('ppo_agent')

