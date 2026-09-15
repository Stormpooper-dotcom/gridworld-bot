import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GridworldEnv(gym.Env):
    def __init__(self, size=10, num_traps=3, use_traps=True, max_steps=100):
        super().__init__()
        self.size = size
        self.num_traps = num_traps if use_traps else 0
        self.use_traps = use_traps
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        
        # 2 for goal dx/dy, 4 for 4 wall distances, 4 for is trap Up/Down/Left/Right = 10
        self.observation_space = spaces.Box(low=-size, high=size, shape=(10,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = (0, 0)
        self.goal = self.agent
        self.current_step = 0
        
        while self.goal == self.agent:
            self.goal = (self.np_random.integers(0, self.size), self.np_random.integers(0, self.size))

        trap_dirs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        self.traps = set()
        
        if self.use_traps:
            while len(self.traps) != self.num_traps:
                tx, ty = self.np_random.integers(0, self.size), self.np_random.integers(0, self.size)
                
                # QUICK CHECK: If we already picked this coordinate, skip immediately
                if (tx, ty) in self.traps:
                    continue
                
                # Check all adjacent directions cleanly before adding
                invalid_spawn = False
                for dirx, diry in trap_dirs:
                    if (tx, ty) == (self.agent[0] + dirx, self.agent[1] + diry) or \
                       (tx, ty) == (self.goal[0] + dirx, self.goal[1] + diry):
                        invalid_spawn = True
                        break
                
                if not invalid_spawn:
                    self.traps.add((tx, ty))
                
        return self._get_obs(), {}

    def _get_obs(self):
        # 14 elements total
        obs_array = np.zeros(10, dtype=np.float32)
        
        # 1. Relative goal vector (2 elements)
        gdx, gdy = self.distance_to(self.goal)
        obs_array[0] = gdx
        obs_array[1] = gdy

        # 2. Distances to walls (4 elements)
        ax, ay = self.agent
        obs_array[2] = ax # Left
        obs_array[3] = self.size-1 - ax # Right
        obs_array[4] = ay # Up
        obs_array[5] = self.size-1 - ay # Down

        # 3. Trap radar (4 elements)
        obs_array[6] = int((ax - 1, ay) in self.traps)
        obs_array[7] = int((ax + 1, ay) in self.traps)
        obs_array[8] = int((ax, ay - 1) in self.traps)
        obs_array[9] = int((ax, ay + 1) in self.traps)

        return obs_array

    def distance_to(self, target_object):
        ax, ay = self.agent
        bx, by = target_object
        return bx-ax, by-ay
    
    def step(self, action):
        self.current_step += 1
        
        # 1. Apply action
        old_x, old_y = self.agent
        x, y = self.agent
        if action == 0:   x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = np.clip(x, 0, self.size-1)
        y = np.clip(y, 0, self.size-1)
        self.agent = (x, y)

        # 2. Determine consequences and rewards
        terminated = False
        truncated = False

        if self.agent == self.goal:
            reward = 100.0
            terminated = True
        elif self.use_traps and (self.agent in self.traps):
            reward = -50.0
            terminated = True
        elif (old_x, old_y) == self.agent:  # Wall bump check
            reward = -10.0
            terminated = False
        else:
            reward = -1.0 # Time discourager
            terminated = False

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render_ascii(self):
        grid = np.full((self.size, self.size), fill_value=".", dtype=str)
        if self.use_traps:
            for tx, ty in self.traps:
                grid[tx, ty] = "X"
        gx, gy = self.goal
        grid[gx, gy] = "G"
        ax, ay = self.agent
        grid[ax, ay] = "A"
        
        print("\n" + "=" * (self.size * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print("=" * (self.size * 2 + 1))
