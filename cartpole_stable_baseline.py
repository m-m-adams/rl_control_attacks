import gymnasium as gym
import PID_cartpole
from stable_baselines3 import PPO

#env = gym.make("CartPole-v1", render_mode='human')
env = PID_cartpole.PIDEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

model.save('trained_a2c_edges')

model.set_env(PID_cartpole.PIDEnv(render_mode='human'))
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
