import gymnasium as gym
from stable_baselines3 import PPO
from ScalarFieldEnv import ScalarFieldEnv
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
np.random.seed(10)

# Load and prepare image
image_path = "N17E073.jpg"
#img = Image.open(image_path).convert("L").resize((256, 256))
img = Image.open(image_path).resize((256, 256))
map_array = np.array(img, dtype=np.uint8)

# Create environment
env = ScalarFieldEnv(map_array, max_steps=1000, render_mode="human")
output = "trainingv30_E23_S50K"
os.makedirs(output, exist_ok=True)
# Set tensorboard log directory
log_dir_root = os.path.join(output, "ppo_scalarfield_tensorboard")
log_name = "PPO_1"  # You can rename this per run
full_log_dir = os.path.join(log_dir_root, log_name)

# Initialize and train PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=full_log_dir, learning_rate=1e-3, n_steps=2000)
model.learn(total_timesteps=60000)

# Evaluate the trained agent
vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        obs = vec_env.reset()


env.close()
model.save(os.path.join(output, "ppo_agent"))

# === Save reward plot from TensorBoard logs ===
# Find the most recent run subdirectory
log_subdirs = [os.path.join(full_log_dir, d) for d in os.listdir(full_log_dir) if os.path.isdir(os.path.join(full_log_dir, d))]
log_subdirs.sort()
if not log_subdirs:
    raise RuntimeError("No subdirectories found in TensorBoard log dir.")
event_log_path = log_subdirs[-1]  # Latest run

# Load TensorBoard logs
event_acc = EventAccumulator(event_log_path)
event_acc.Reload()

# Extract reward data
rewards = event_acc.Scalars('rollout/ep_rew_mean')
steps = [e.step for e in rewards]
values = [e.value for e in rewards]

# Plot and save
plt.figure(figsize=(10, 5))
plt.plot(steps, values, label='rollout/ep_rew_mean', color='blue')
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("PPO Training Performance")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output,"ppo_ep_rew_mean_plot.png"))  # or use .svg
plt.close()

print("✅ Saved reward plot as 'ppo_ep_rew_mean_plot.png'")
