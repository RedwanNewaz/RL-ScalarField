from custom_environment import ImageExplorationEnv
from PIL import Image
import numpy as np
from time import sleep


image_path = "N17E073.jpg"
img = Image.open(image_path).convert("L").resize((256, 256))
map_array = np.array(img, dtype=np.uint8)

env = ImageExplorationEnv(map_array, max_steps=700, render_mode="human")


for epoch in range (10):
        
    # Manually preserve visited map
    total_steps = 0
    done= False
    obs = env.reset()
    explored = env.visited.copy()
    episode_reward= 0
    while not done:
        action = env.action_space.sample()
        
        obs, reward, done, _, _ = env.step(action)
        episode_reward+= reward
        env.render(scale_factor=3)
        #sleep(0.1)
        total_steps += 1
        print(obs.shape)

        
    print("Exploration complete.")
    print(episode_reward )
env.close()
