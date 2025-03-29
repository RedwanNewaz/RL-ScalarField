import gym
from gym import spaces
import numpy as np
import cv2

class ImageExplorationEnv(gym.Env):
    def __init__(self, map_array: np.ndarray, max_steps: int = 1000, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.base_map = map_array.astype(np.uint8)
        assert self.base_map.shape == (256, 256), "Map must be 256x256 pixels"
        self.height, self.width = self.base_map.shape
        self.grid_size = self.width // 32
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.visited = np.zeros((self.height, self.width), dtype=bool)
        self.agent_pos = None
        self.max_steps = max_steps
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.agent_pos = (np.random.randint(0, 32) * self.grid_size,
                          np.random.randint(0, 32) * self.grid_size)
        self.visited.fill(False)
        
        return self._get_observation(), {}

    def step(self, action):
        r, c = self.agent_pos
        if action == 0:    # Up
            r = max(r - self.grid_size, 0)
        elif action == 1:  # Down
            r = min(r + self.grid_size, self.height - self.grid_size)
        elif action == 2:  # Left
            c = max(c - self.grid_size, 0)
        elif action == 3:  # Right
            c = min(c + self.grid_size, self.width - self.grid_size)
        
        self.agent_pos = (r, c)
        newly_visited = not np.all(self.visited[r:r+self.grid_size, c:c+self.grid_size])
        self.visited[r:r+self.grid_size, c:c+self.grid_size] = True
        
        reward = np.mean(self.base_map[r:r+self.grid_size, c:c+self.grid_size]) if newly_visited else -1
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps or np.all(self.visited)
        
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        image=(self.base_map * self.visited.astype(np.uint8))[:, :, None]
        return np.squeeze(image)

    def render(self, mode=None, scale_factor=3):
        if mode is None:
            mode = self.render_mode

        grid_image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        for i in range(0, self.height, self.grid_size):
            cv2.line(grid_image, (0, i), (self.width, i), (0,0,0), 1)
        for i in range(0, self.width, self.grid_size):
            cv2.line(grid_image, (i, 0), (i, self.height), (0,0,0), 1)

        obs_grid = self.base_map * self.visited.astype(np.uint8)
        color_img = cv2.cvtColor(obs_grid, cv2.COLOR_GRAY2BGR)
        blended_image = np.where(self.visited[..., None], color_img, grid_image)

        r, c = self.agent_pos
        cv2.rectangle(blended_image, (c, r), (c+self.grid_size, r+self.grid_size), (255,0,0), -1)

        if mode == "human":
            resized = cv2.resize(blended_image, (self.width * scale_factor, self.height * scale_factor), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Exploration", resized)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return blended_image

    def close(self):
        cv2.destroyAllWindows()
