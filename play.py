from gym.utils.play import play
from custom_environment import ImageExplorationEnv
from PIL import Image
import numpy as np

image_path = "N17E073.jpg"
img = Image.open(image_path).convert("L").resize((256, 256))
map_array = np.array(img, dtype=np.uint8)

env = ImageExplorationEnv(map_array, max_steps=200, render_mode="rgb_array")

keys_to_actions = {
    (ord("w"),): 0,  # Move up
    (ord("s"),): 1,  # Move down
    (ord("a"),): 2,  # Move left
    (ord("d"),): 3,  # Move right
}

play(env, keys_to_action=keys_to_actions, zoom=3.0)
env.close()
