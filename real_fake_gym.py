# %%
import cartpole
import gymnasium as gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, state, path='./images/', filename='gym_animation.gif', goals=None):


    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
               frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    position  = plt.text(0, 0, f'Position {0} \nAngle {0}', bbox=dict(fill=False, edgecolor='red', linewidth=2))

    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        print(state[i])
        position.set_text(f'Position {state[i][0]: 4.2f} \nAngle    {state[i][2]: 4.2f}')
        if goals:
            position.set_text(f'Position {state[i][0]: 4.2f} \nAngle    {state[i][2]: 4.2f} \nCommand  {goals[i]: 4.2f}')


    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames))
    anim.save(path + filename, fps=30)


class PD:
    def __init__(self, kp, kd, goal):
        self.kp = kp
        self.kd = kd
        self.goal = goal
        self.last_error = 0

    def observe(self, x):
        error = self.goal - x
        d_error = error - self.last_error
        self.last_error = error
        return self.kp * error + self.kd * d_error


class PIDEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.controller_theta = PD(5, 100, 0)
        self.controller_x = PD(5, 50, 0)
        self.action_space = gym.spaces.Box(
            low=-.1, high=.1, shape=(1,), dtype=np.float32)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        self.controller_x.goal = action
        x, x_dot, theta, theta_dot = self.state
        print(x, theta)
        theta_action = self.controller_theta.observe(theta)
        x_action = self.controller_x.observe(x)
        act = 1 if theta_action + x_action < 0 else 0
        state, reward, terminated, truncated, _ = self.env.step(act)
        reward = 1-np.abs(1-state[0])
        return state, reward, terminated, truncated, _


if __name__ == '__main__':
    frames = []
    states = []
    goals = []
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    env = PIDEnv(env)
    state, _ = env.reset()
    for i in range(500):
        if i < 10:
            step = 0
        elif i < 250:
            step = 1
        elif i < 500:
            step = 0
        print(step)
        state, reward, terminated, truncated, _ = env.step(np.float32(step))
        frames.append(env.render())
        states.append(state)
        goals.append(step)
        if terminated:
            print('fell')
            state = env.reset()

    save_frames_as_gif(frames, states, filename='PID_centering_test.gif', goals=goals)

    
