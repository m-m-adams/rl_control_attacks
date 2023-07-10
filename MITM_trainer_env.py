# %%
import cartpole
import gymnasium as gym
import numpy as np
from PID_cartpole import PD
import matplotlib.pyplot as plt


class PIDEnvTrainer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.controller_theta = PD(5, 100, 0)
        self.controller_x = PD(1, 50, 0)
        self.goal = 0
        self.steps = 0
        self.action_space = gym.spaces.Box(
            low=-.1, high=.1, shape=(1,), dtype=np.float32)

    def reset(self):
        return self.env.reset()

    #action represents a sensor modification
    def step(self, action):
        #provide input to the PID control layer
        if self.steps%100 == 0:
            self.goal += np.random.randint(-1,1)
            if self.goal > 2 or self.goal < -2:
                self.goal = 0
            self.controller_x.goal = self.goal
    
        x, x_dot, theta, theta_dot = self.state
        theta_action = self.controller_theta.observe(action.item()+theta)

        x_action = self.controller_x.observe(x)
        act = 1 if theta_action + x_action < 0 else 0
        state, reward, terminated, truncated, _ = self.env.step(act)
        speed = state[1]

        reward += 1 if abs(speed) >0.5 else 0

        self.steps += 1

        return state, reward, terminated, truncated, _



