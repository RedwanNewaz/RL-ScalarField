import gymnasium as gym
from stable_baselines3 import PPO
from ScalarFieldEnv import ScalarFieldEnv
from PIL import Image
import os
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)

    # load image
    image_path = cfg.map.file_path
    resolution = cfg.map.resolution
    img = Image.open(image_path).resize(resolution)
    map_array = np.array(img, dtype=np.uint8)

    # Create environment
    env = ScalarFieldEnv(map_array,
                   num_square_cells=cfg.gym.num_square_cells,
                   max_steps=cfg.gym.max_steps,
                   render_mode=cfg.gym.render_mode,
                   epsilon=cfg.gym.epsilon
                   )
    output = cfg.output
    os.makedirs(output, exist_ok=True)

    # Set tensorboard log directory
    log_dir_root = os.path.join(output, "ppo_scalarfield_tensorboard")
    log_name = "PPO_1"  # You can rename this per run
    full_log_dir = os.path.join(log_dir_root, log_name)

    # Initialize and train PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=full_log_dir, learning_rate=1e-3, n_steps=2000)
    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(os.path.join(output, "ppo_agent"))

    # === Save reward plot from TensorBoard logs ===
    # Find the most recent run subdirectory
    log_subdirs = [os.path.join(full_log_dir, d) for d in os.listdir(full_log_dir) if
                   os.path.isdir(os.path.join(full_log_dir, d))]
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
    plt.savefig(os.path.join(output, "ppo_ep_rew_mean_plot.png"))  # or use .svg
    plt.close()
    print("✅ Saved reward plot as 'ppo_ep_rew_mean_plot.png'")

if __name__ == "__main__":
    main()
