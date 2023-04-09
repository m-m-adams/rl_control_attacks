# %%
import skrl_angle_intercept as trained
import PID_cartpole
import torch
import gymnasium as gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from skrl.envs.torch import wrap_env


class MITMEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agent = trained.agent_ddpg
        self.agent.init()
        self._env = wrap_env(env)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, terminated, truncated, _ = self.env.step(
            action)
        modification = self.agent.act(torch.from_numpy(
            state).to(trained.device), 1000, 10000)[0]
        reward = 1-np.abs(1-state[0])
        print(modification)
        state[2] = state[2]+modification

        return state, reward, terminated, truncated, _


if __name__ == '__main__':
    frames = []
    with torch.no_grad():
        env = gym.make("CartPole-v1", render_mode='human')
        env = MITMEnv(env)
        env = PID_cartpole.PIDEnv(env)
        env = wrap_env(env)
        state, _ = env.reset()

        for i in range(500):
            state, reward, terminated, truncated, _ = env.step(torch.zeros(1,))
            frames.append(env.render())
            if terminated:
                print('fell')
                state = env.reset()

    env.close()
