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
    def __init__(self, env, agent_path, noise_path=None):
        super().__init__(env)
        self._env = wrap_env(env)
        self.action_space = gym.spaces.Discrete(2)
        self.real_state = self.state
        self.attack = False
        self.steps = 0
        mitm_mod_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
        try:
            self.agent = trained.load(mitm_mod_space
                                  , self.observation_space, file=agent_path)
            self.add_actions = False
        except RuntimeError:
            high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                2.,
            ],
            dtype=np.float32,
            )
            observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
            self.agent = trained.load(mitm_mod_space, observation_space, file=agent_path)
            self.add_actions = True
        if noise_path:
            self.noise_agent = trained.load(mitm_mod_space, self.observation_space, file=noise_path)


    def reset(self):
        return self.env.reset()

    def step(self, action):
        if self.steps%50==0:
            self.attack = np.random.choice([True, False], p=[0.2,0.8])
        

        self.state = self.real_state
        state, reward, terminated, truncated, _ = self.env.step(action)
        self.real_state = state
        new_state = state
        if self.attack:
            if self.add_actions:
                new_state = np.append(state, action)

            modification = self.agent.act(torch.from_numpy(
                new_state).to(trained.device, dtype=torch.float32), 1000, 10000)[0]
        else:
            modification = self.noise_agent.act(torch.from_numpy(
        state).to(trained.device, dtype=torch.float32), 1000, 10000)[0]
        reward = 1-np.abs(1-state[0])
        state[2] = state[2]+modification
        self.state = state
        self.steps+=1


        return np.append(state, self.attack), reward, terminated, truncated, _
    


