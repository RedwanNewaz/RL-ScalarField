from stable_baselines3 import PPO
from ScalarFieldEnv import ScalarFieldEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from PIL import Image
import os
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # load image
    image_path = cfg.map.file_path
    resolution = cfg.map.resolution
    img = Image.open(image_path).resize(resolution)
    map_array = np.array(img, dtype=np.uint8)

    # Wrap your environment
    env = ScalarFieldEnv(cfg.robot,
                          map_array,
                          num_square_cells=cfg.gym.num_square_cells,
                          max_steps=cfg.gym.max_steps,
                          render_mode=cfg.gym.render_mode,
                          epsilon=cfg.gym.epsilon
                          )

    # load trajectory
    traj_path = cfg.benchmark.npy_file
    traj = np.load(traj_path)
    #traj = np.load("results/trainingv1n35w107/trajectories/episode_003_trajectory.npy") - 31.0

    # normalize traj
    traj = (traj + cfg.gym.num_square_cells / 2.0)

    assert np.min(traj) >= 0, f"Trajectory contains negative values: {np.min(traj)}"
    traj = traj[:cfg.gym.max_steps, :]
    traj = traj.astype(np.uint8)

    rewards = [env.get_reward(x, y) for x, y in traj]
    print(f"Trajectory shape: {traj.shape}, Total reward {np.sum(rewards):.4f} Step reward mean {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

if __name__ == "__main__":
    main()