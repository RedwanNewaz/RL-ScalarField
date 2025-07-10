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
    env = DummyVecEnv([lambda: ScalarFieldEnv(cfg.robot,
                                              map_array,
                                              num_square_cells=cfg.gym.num_square_cells,
                                              max_steps=cfg.gym.max_steps,
                                              render_mode=cfg.gym.render_mode,
                                              epsilon=cfg.gym.epsilon
                                              )])

    # Load the trained agent
    output = cfg.output
    os.makedirs(output, exist_ok=True)
    model_path = os.path.join(output, "ppo_agent.zip")
    model = PPO.load(model_path)

    # Create trajectory directory
    trajectory_dir = os.path.join(output, "trajectories")
    os.makedirs(trajectory_dir, exist_ok=True)

    # Run it
    if cfg.visualize:
        obs = env.reset()
        episode = 0
        max_episodes = cfg.n_eval_episodes
        episode_reward = 0.0
        trajectory = []
        while episode < max_episodes:

            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            trajectory.append(list(info[0]['position']))

            env.render()
            episode_reward += rewards[0]
            if dones:
                obs = env.reset()
                print(f"Episode {episode} finished with episode reward: {episode_reward:.4f}")
                #print(trajectory)
                trajectory_array = np.array(trajectory)
                trajectory_file = os.path.join(trajectory_dir, f"episode_{episode + 1:03d}_trajectory.npy")
                np.save(trajectory_file, trajectory_array)
                print(f"Trajectory shape: {trajectory_array.shape}")

                #print(f"Episode {episode} finished with episode reward: {episode_reward:.4f}")
                episode += 1
                episode_reward = 0.0
                trajectory = []

    else:
        print("Evaluating the agent {} for {} episodes. This will take some times ..".format(output, cfg.n_eval_episodes))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=cfg.n_eval_episodes)
        print(f"Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

if __name__ == "__main__":
    main()