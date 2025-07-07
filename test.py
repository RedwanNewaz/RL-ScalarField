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

    # Run it
    if cfg.visualize:
        obs = env.reset()
        episode = 0
        max_episodes = cfg.n_eval_episodes
        total_reward = 0.0
        while episode < max_episodes:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            total_reward += rewards[0]
            if dones:
                episode += 1
                obs = env.reset()
                print(f"Episode {episode} finished with avg reward: {total_reward / max_episodes:.4f}")
                total_reward = 0.0
    else:
        print("Evaluating the agent {} for {} episodes. This will take some times ..".format(output, cfg.n_eval_episodes))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=cfg.n_eval_episodes)
        print(f"Mean reward: {mean_reward:.4f} +/- {std_reward:.4f}")

if __name__ == "__main__":
    main()