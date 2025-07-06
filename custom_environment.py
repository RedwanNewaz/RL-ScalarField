import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from sensors.point_sensor import PointSensor


class ImageExplorationEnv(gym.Env):
    def __init__(self, map_array: np.ndarray, max_steps: int = 1000, render_mode="rgb_array", use_sensor=False,
                 sensor_rate=1.0, noise_scale=1.0):

        super().__init__()
        self.render_mode = render_mode
        self.base_map = map_array.astype(np.uint8)
        assert self.base_map.shape == (256, 256), "Map must be 256x256 pixels"
        self.height, self.width = self.base_map.shape
        self.grid_size = self.width // 32

        self.use_sensor = use_sensor
        if use_sensor:
            env_extent = [0, self.width, 0, self.height]
            rng = np.random.RandomState(42)
            self.sensor = PointSensor(map_array, env_extent=env_extent, rate=sensor_rate, noise_scale=noise_scale,
                                      rng=rng)

        self.delta_t = 1.0
        self.linear_velocities = np.array([0.25, 0.5, 0.75, 1.0])
        self.angular_velocities = np.array([-0.7, -0.35, 0.0, 0.35, 0.7])
        self.action_space = spaces.MultiDiscrete([len(self.linear_velocities), len(self.angular_velocities)])

        obs_length = 2 + 1 + 256 * 256
        if use_sensor:
            obs_length += 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_length,), dtype=np.float32)

        self.visited = np.zeros((self.height, self.width), dtype=bool)
        self.agent_pos = None
        self.agent_orientation = None
        self.max_steps = max_steps
        self.steps_taken = 0
        self.episode_reward = 0

        self.visit_count = np.zeros((32, 32), dtype=int)
        self.last_positions = []
        self.position_history_length = 40
        self.last_actions = []
        self.action_history_length = 20
        self.straight_line_bonus_counter = 0
        self.last_grid_pos = None
        self.unique_cells_visited = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0

        margin = 3
        grid_x = np.random.randint(margin, 32 - margin)
        grid_y = np.random.randint(margin, 32 - margin)
        self.agent_pos = np.array([grid_x * self.grid_size + self.grid_size // 2,
                                   grid_y * self.grid_size + self.grid_size // 2], dtype=np.float32)
        self.agent_orientation = np.random.uniform(0, 2 * np.pi)

        self.visited.fill(False)
        self.visit_count.fill(0)
        self.last_positions = []
        self.last_actions = []
        self.straight_line_bonus_counter = 0
        self.last_grid_pos = None
        self.episode_reward = 0
        self.unique_cells_visited = set()

        self._mark_visited()
        return self._get_observation(), {}

    def step(self, action):
        linear_vel_idx, angular_vel_idx = action
        linear_velocity = self.linear_velocities[linear_vel_idx]
        angular_velocity = self.angular_velocities[angular_vel_idx]

        self.last_actions.append(action)
        if len(self.last_actions) > self.action_history_length:
            self.last_actions.pop(0)

        self.agent_orientation = (self.agent_orientation + angular_velocity * self.delta_t) % (2 * np.pi)
        dx = linear_velocity * np.cos(self.agent_orientation) * self.delta_t
        dy = linear_velocity * np.sin(self.agent_orientation) * self.delta_t
        new_pos = self.agent_pos + np.array([dx, dy])

        wall_hit = False
        if new_pos[0] < self.grid_size // 2:
            new_pos[0] = self.grid_size // 2
            self.agent_orientation = np.pi - self.agent_orientation
            wall_hit = True
        elif new_pos[0] > self.width - self.grid_size // 2:
            new_pos[0] = self.width - self.grid_size // 2
            self.agent_orientation = np.pi - self.agent_orientation
            wall_hit = True

        if new_pos[1] < self.grid_size // 2:
            new_pos[1] = self.grid_size // 2
            self.agent_orientation = -self.agent_orientation
            wall_hit = True
        elif new_pos[1] > self.height - self.grid_size // 2:
            new_pos[1] = self.height - self.grid_size // 2
            self.agent_orientation = -self.agent_orientation
            wall_hit = True

        self.agent_orientation %= (2 * np.pi)
        self.agent_pos = new_pos
        newly_visited = self._mark_visited()

        grid_x = int(self.agent_pos[0] // self.grid_size)
        grid_y = int(self.agent_pos[1] // self.grid_size)
        grid_x = np.clip(grid_x, 0, 31)
        grid_y = np.clip(grid_y, 0, 31)
        r = grid_y * self.grid_size
        c = grid_x * self.grid_size
        current_grid_pos = (grid_x, grid_y)

        self.last_positions.append(current_grid_pos)
        if len(self.last_positions) > self.position_history_length:
            self.last_positions.pop(0)

        if self.last_grid_pos != current_grid_pos:
            self.straight_line_bonus_counter += 1
        else:
            self.straight_line_bonus_counter = 0
        self.last_grid_pos = current_grid_pos

        patch = self.base_map[r:r + self.grid_size, c:c + self.grid_size] / 255.0
        mean_pixel_value = np.mean(patch)
        visits = self.visit_count[grid_y, grid_x]

        # Compute rewards
        pixel_reward = 50.0 * (1 / (1 + np.exp(-12 * (mean_pixel_value - 0.5))) - 0.5)
        revisit_penalty = -1.0 * (visits ** 2) if visits > 1 else 0.0
        exploration_bonus = 1.0 if current_grid_pos not in self.unique_cells_visited else 0.0
        wall_penalty = -1.0 if wall_hit else 0.0
        step_cost = -0.05

        # Final reward
        reward = pixel_reward + revisit_penalty + exploration_bonus + wall_penalty + step_cost

        # Update visit history
        if current_grid_pos not in self.unique_cells_visited:
            self.unique_cells_visited.add(current_grid_pos)

        # === Generalized Reward Function with Separate Revisit Penalty ===
        # if visits > 1:
        #     reward = -2 * (visits ** 2)
        #     #reward = -np.exp(visits * 0.2) + 1
        #
        # else:
        #     reward = 40.0 * (1 / (1 + np.exp(-8 * (mean_pixel_value - 0.5))) - 0.5)
        #
        # #base_reward = 4.0 * (1 / (1 + np.exp(-8 * (mean_pixel_value - 0.5))) - 0.5)
        # #revisit_penalty = -0.5 * np.log1p(visits) if visits > 1 else 0.0
        # #reward = base_reward + revisit_penalty
        #
        # if current_grid_pos not in self.unique_cells_visited:
        #     reward += 0.5
        #     self.unique_cells_visited.add(current_grid_pos)
        #
        # if wall_hit:
        #     reward -= 1.0

        #reward -= 0.01
        #reward = np.clip(reward, -3.0, 3.0)
        self.episode_reward += reward

        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps or np.all(self.visited)

        if self.use_sensor:
            print("Sensor reading:",
                  self.sensor.sense(np.array([self.agent_pos[0], self.agent_pos[1], self.agent_orientation])))

        return self._get_observation(), reward, done, False, {}

    def _mark_visited(self):
        grid_x = int(self.agent_pos[0] // self.grid_size)
        grid_y = int(self.agent_pos[1] // self.grid_size)
        grid_x = np.clip(grid_x, 0, 31)
        grid_y = np.clip(grid_y, 0, 31)
        r = grid_y * self.grid_size
        c = grid_x * self.grid_size
        newly_visited = not np.all(self.visited[r:r + self.grid_size, c:c + self.grid_size])
        self.visited[r:r + self.grid_size, c:c + self.grid_size] = True
        self.visit_count[grid_y, grid_x] += 1
        return newly_visited

    def _get_observation(self):
        image = (self.base_map * self.visited.astype(np.uint8))[:, :, None]
        obs = np.squeeze(image) / 255.0
        obs = obs.flatten()
        agent_pos_normalized = self.agent_pos / self.width
        orientation_normalized = self.agent_orientation / (2 * np.pi)
        obs = np.hstack((agent_pos_normalized, [orientation_normalized], obs))

        if self.use_sensor:
            sensor_obs = self.sensor.sense(np.array([self.agent_pos[0], self.agent_pos[1], self.agent_orientation]))
            obs = np.hstack((obs, sensor_obs.flatten()))
        return obs

    def render(self, mode=None, scale_factor=3):
        if mode is None:
            mode = self.render_mode
        grid_image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        for i in range(0, self.height, self.grid_size):
            cv2.line(grid_image, (0, i), (self.width, i), (0, 0, 0), 1)
        for i in range(0, self.width, self.grid_size):
            cv2.line(grid_image, (i, 0), (i, self.height), (0, 0, 0), 1)
        obs_grid = self.base_map * self.visited.astype(np.uint8)
        color_img = cv2.cvtColor(obs_grid, cv2.COLOR_GRAY2BGR)
        blended_image = np.where(self.visited[..., None], color_img, grid_image)
        agent_pixel_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        cv2.circle(blended_image, agent_pixel_pos, self.grid_size // 4, (255, 0, 0), -1)
        arrow_length = self.grid_size // 2
        end_x = int(self.agent_pos[0] + arrow_length * np.cos(self.agent_orientation))
        end_y = int(self.agent_pos[1] + arrow_length * np.sin(self.agent_orientation))
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
