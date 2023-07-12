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
    def __init__(self, env):
        super().__init__(env)
        self.controller_theta = PD(5, 100, 0)
        self.controller_x = PD(.75, 75, 0, 0.5)
        self.goal = 0
        self.steps = 0
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)
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
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)


    def reset(self):
        self.steps = 0
        state, info = self.env.reset()
        state = np.append(state, 0)
        return state, info

    #action represents a sensor modification
    def step(self, action):
        #provide input to the PID control layer
        if self.steps%100 == 0:
            self.goal = np.random.randint(-2,3)
            self.controller_x.goal = self.goal
    
        x, x_dot, theta, theta_dot = self.state
        theta_action = self.controller_theta.observe(theta-action.item())
        x_action = self.controller_x.observe(x)
        act = 1 if theta_action + x_action < 0 else 0
        state, reward, terminated, truncated, _ = self.env.step(act)
        speed = state[1]
        position = state[0]
        state = np.append(state, self.goal)
        position_err = (self.goal - position)

        #reward = 10-5*small_err_func(position_err*5)-10*abs(action.item())
        reward = 10-10*(action.item()**2)

        self.steps += 1
        if terminated and self.steps < 500:
            reward = -25

        if truncated:
            reward += 50

        return state, reward, terminated, truncated, _

