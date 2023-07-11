# %%
import cartpole
import gymnasium as gym
import numpy as np
from PID_cartpole import PD
import matplotlib.pyplot as plt

#This function is 1 around zero and for high x, 0 at small x
#basically two dips around x=1 and -1
def small_err_func(x):
    return (x**4 - 2*(x**2) + 1) / ((x**4) + 1)

class PIDEnvTrainer(gym.Wrapper):
    def __init__(self, env, pretrain=False):
        super().__init__(env)
        self.controller_theta = PD(5, 100, 0)
        self.controller_x = PD(.1, 50, 0)
        self.goal = 0
        self.steps = 0
        self.pretrain = pretrain
        self.action_space = gym.spaces.Box(
            low=-.01, high=.01, shape=(1,), dtype=np.float32)

    def reset(self):
        self.steps = 0
        return self.env.reset()

    #action represents a sensor modification
    def step(self, action):
        #provide input to the PID control layer
        if self.steps%100 == 0:
            self.goal = np.random.randint(-1,1)
            self.controller_x.goal = self.goal
    
        x, x_dot, theta, theta_dot = self.state
        theta_action = self.controller_theta.observe(theta-action.item())

        x_action = self.controller_x.observe(x)
        act = 1 if theta_action + x_action < 0 else 0
        state, reward, terminated, truncated, _ = self.env.step(act)
        speed = state[1]
        position = state[0]
        position_err = (self.goal - position)

        reward = 10-5*small_err_func(position_err*5)-10*abs(action.item())
        #reward = 10-10*abs(action.item())

        self.steps += 1
        if terminated and self.steps < 500:
            reward = -25

        return state, reward, terminated, truncated, _

