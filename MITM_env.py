# %%
import train_and_view as trained
import PID_cartpole
import torch
import gymnasium as gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from skrl.envs.torch import wrap_env


class MITMEnv(gym.Wrapper):
    def __init__(self, env, agent_path):
        super().__init__(env)
        self.agent = trained.load(file=agent_path)
        self.agent.init()
        self._env = wrap_env(env)
        self.action_space = gym.spaces.Discrete(2)
        self.real_state = self.state

    def reset(self):
        return self.env.reset()

    def step(self, action):
        self.state = self.real_state
        state, reward, terminated, truncated, _ = self.env.step(
            action)
        self.real_state = state
        modification = self.agent.act(torch.from_numpy(
            state).to(trained.device), 1000, 10000)[0]
        reward = 1-np.abs(1-state[0])
        state[2] = state[2]+modification
        self.state = state

        return state, reward, terminated, truncated, _



