import gymnasium as gym
import math
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
import cv2


@dataclass
class RobotConfig:
    linear_velocities = [0.25, 0.5, 0.75, 1.0]
    angular_velocities = [-0.7, -0.35, 0.0, 0.35, 0.7]
    dt: float = 0.5
    obs_length: int = 2 + 1 + 256 * 256


class Robot:
    def __init__(self, config: RobotConfig, grid_bounds: int = 32):
        self.config = config
        self.grid_bounds = grid_bounds
        self.state = np.zeros((4, 1))  # x, y, theta, v

    def reset(self, position):
        # x, y, theta, v
        self.state = np.array([[position[0]], [position[1]], [0.0], [0.0]], dtype=np.float32)
        return self.state

    def predict(self, action):
        v_indx, w_indx = action
        linear_velocity = self.config.linear_velocities[v_indx]
        angular_velocity = self.config.angular_velocities[w_indx]
        u = np.array([[linear_velocity], [angular_velocity]], dtype=np.float32)
        state = self.motion_model(self.state, u, self.config.dt)
        return np.squeeze(self.observation_model(state))

    def update(self, action):
        v_indx, w_indx = action
        linear_velocity = self.config.linear_velocities[v_indx]
        angular_velocity = self.config.angular_velocities[w_indx]
        u = np.array([[linear_velocity], [angular_velocity]], dtype=np.float32)
        self.state = self.motion_model(self.state, u, self.config.dt)
        # Clip to stay within bounds (0 to grid_bounds-1)
        self.state[0, 0] = np.clip(self.state[0, 0], 0, self.grid_bounds - 1)
        self.state[1, 0] = np.clip(self.state[1, 0], 0, self.grid_bounds - 1)
        self.state[2, 0] = self.state[2, 0] % (2 * np.pi)

    @staticmethod
    def motion_model(x, u, dt):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[dt * math.cos(x[2, 0]), 0],
                      [dt * math.sin(x[2, 0]), 0],
                      [0.0, dt],
                      [1.0, 0.0]])

        x = F @ x + B @ u

        return x

    @staticmethod
    def observation_model(x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H @ x

        return z


class ScalarFieldEnv(gym.Env):
    def __init__(self, map_array: np.ndarray, max_steps: int = 1000, num_square_cells: int = 32,
                 render_mode: str = "human", epsilon: float = 0.99):
        super().__init__()
        self.base_map = map_array.astype(np.uint8)
        assert self.base_map.shape == (256, 256), "Map must be 256x256 pixels"
        self.height, self.width = self.base_map.shape
        self.grid_size = self.width // num_square_cells
        self.max_steps = max_steps
        # Fixed: Initialize visit_count with correct dimensions (32x32 grid)
        self.visit_count = np.zeros((num_square_cells, num_square_cells), dtype=int)
        self.__robot_config = RobotConfig()
        # Fixed: Pass grid bounds to robot
        self.__robot = Robot(self.__robot_config, num_square_cells)
        self.num_square_cells = num_square_cells
        self.__step_taken = 0
        self.render_mode = render_mode
        self.epsilon = epsilon

        # gym spaces
        self.action_space = spaces.MultiDiscrete(
            [len(self.__robot_config.linear_velocities), len(self.__robot_config.angular_velocities)])

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.__robot_config.obs_length,), dtype=np.float32)

    def _get_observation(self):
        # Create visited map at pixel level for visualization
        visited_pixels = np.zeros((self.height, self.width), dtype=bool)

        # Mark visited grid cells at pixel level
        for i in range(self.num_square_cells):
            for j in range(self.num_square_cells):
                if self.visit_count[i, j] > 0:
                    # Mark the entire grid cell as visited
                    y_start = i * self.grid_size
                    y_end = (i + 1) * self.grid_size
                    x_start = j * self.grid_size
                    x_end = (j + 1) * self.grid_size
                    visited_pixels[y_start:y_end, x_start:x_end] = True

        # Apply base map to visited areas
        image = (self.base_map * visited_pixels.astype(np.uint8))[:, :, None]
        obs = np.squeeze(image) / 255.0
        obs = obs.flatten()

        # Normalize robot position and orientation
        x = self.__robot.state[0, 0] / self.num_square_cells
        y = self.__robot.state[1, 0] / self.num_square_cells
        theta = self.__robot.state[2, 0] / (2 * np.pi)

        obs = np.hstack(([x, y, theta], obs))
        return obs

    def reset(self, seed=1234, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        margin = 3
        grid_x = np.random.randint(margin, self.num_square_cells - margin)
        grid_y = np.random.randint(margin, self.num_square_cells - margin)

        # Fixed: Reset visit count and mark initial position
        self.visit_count = np.zeros((self.num_square_cells, self.num_square_cells), dtype=int)
        self.visit_count[grid_y, grid_x] = 1
        self.__step_taken = 0

        self.__robot.reset(position=(grid_x, grid_y))
        return self._get_observation(), {}

    def step(self, action):
        self.__robot.update(action)
        x, y, _, _ = self.__robot.state.flatten()

        # Fixed: Convert continuous position to grid coordinates properly
        grid_x = int(np.clip(x, 0, self.num_square_cells - 1))
        grid_y = int(np.clip(y, 0, self.num_square_cells - 1))

        # Update visit count for the grid cell
        self.visit_count[grid_y, grid_x] += 1

        obs = self._get_observation()
        self.__step_taken += 1

        epsilon = self.epsilon / self.__step_taken
        reward = self.base_map[grid_y, grid_x] / 255.0  + epsilon / np.sqrt(np.log(self.visit_count[grid_y, grid_x] + 1))


        # 1. Penalize staying too long in same cell
        if self.visit_count[grid_y, grid_x] > 5:
            reward -= 0.5


        done = self.__step_taken >= self.max_steps
        info = {
            'grid_position': (grid_x, grid_y),
            'visit_count': self.visit_count[grid_y, grid_x],
            'total_visited_cells': np.sum(self.visit_count > 0)
        }
        return obs, reward, done, False, info

    def render(self, mode="human", scale_factor=3):
        if mode is None:
            mode = self.render_mode

        # Create base image
        grid_image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Draw grid lines
        for i in range(0, self.height, self.grid_size):
            cv2.line(grid_image, (0, i), (self.width, i), (0, 0, 0), 1)
        for i in range(0, self.width, self.grid_size):
            cv2.line(grid_image, (i, 0), (i, self.height), (0, 0, 0), 1)

        # Create visited overlay
        visited_pixels = np.zeros((self.height, self.width), dtype=bool)

        # Mark visited grid cells
        for i in range(self.num_square_cells):
            for j in range(self.num_square_cells):
                if self.visit_count[i, j] > 0:
                    y_start = i * self.grid_size
                    y_end = (i + 1) * self.grid_size
                    x_start = j * self.grid_size
                    x_end = (j + 1) * self.grid_size
                    visited_pixels[y_start:y_end, x_start:x_end] = True

        # Apply base map to visited areas
        obs_grid = self.base_map * visited_pixels.astype(np.uint8)
        color_img = cv2.cvtColor(obs_grid, cv2.COLOR_GRAY2BGR)
        blended_image = np.where(visited_pixels[..., None], color_img, grid_image)

        # Draw robot position
        pose = self.__robot.state.flatten()[:3]
        # Fixed: Convert grid coordinates to pixel coordinates
        agent_pixel_x = int(pose[0] * self.grid_size + self.grid_size // 2)
        agent_pixel_y = int(pose[1] * self.grid_size + self.grid_size // 2)
        agent_pixel_pos = (agent_pixel_x, agent_pixel_y)

        # Draw robot as circle
        cv2.circle(blended_image, agent_pixel_pos, self.grid_size // 4, (255, 0, 0), -1)

        # Draw direction arrow
        arrow_length = self.grid_size // 2
        end_x = int(agent_pixel_x + arrow_length * np.cos(pose[2]))
        end_y = int(agent_pixel_y + arrow_length * np.sin(pose[2]))
        cv2.arrowedLine(blended_image, agent_pixel_pos, (end_x, end_y), (0, 0, 255), 2)

        if mode == "human":
            resized = cv2.resize(blended_image, (self.width * scale_factor, self.height * scale_factor),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Exploration", resized)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return blended_image

    def close(self):
        cv2.destroyAllWindows()