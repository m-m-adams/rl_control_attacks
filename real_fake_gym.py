import gym
from gym import spaces
import PID_cartpole
import MITM_env
import numpy as np


class Maybe_MITM(gym.Wrapper):
    def __init__(self, env, mitm_chance = 0.1):
        super().__init__(env)
        self.mitm_chance = mitm_chance


    def reset(self):
        if np.random.random_sample() > self.mitm_chance:
            self.env = PID_cartpole.PIDEnv(self.unwrapped)
        else:
            print('mitm')
            self.env = PID_cartpole.PIDEnv(MITM_env.MITMEnv(self.unwrapped))
        return self.env.reset()


if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode='human')
    env = Maybe_MITM(env, mitm_chance=0.5)
    state, _ = env.reset()
    for i in range(1000):
        state, reward, terminated, truncated, _ = env.step(np.float32(0))
        if terminated:
            print('fell')
            state = env.reset()
        if i%200 == 0:
            state = env.reset()