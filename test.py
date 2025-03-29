import gymnasium as gym

from stable_baselines3 import PPO
from custom_environment import ImageExplorationEnv
from PIL import Image
import numpy as np
from time import sleep
#from stable_baselines import DQN
from stable_baselines3.common.evaluation import evaluate_policy

image_path = "N17E073.jpg"
img = Image.open(image_path).convert("L").resize((256, 256))
map_array = np.array(img, dtype=np.uint8)
env = ImageExplorationEnv(map_array, max_steps=700, render_mode="human")


# Load the trained agent
model = PPO.load("ppo_agent")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()