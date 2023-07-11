import gymnasium as gym
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import MITM_trainer_env
import MITM_env
import PID_cartpole
# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dstr = 'cpu'
if torch.cuda.is_available():
    dstr = 'cuda'
elif torch.backends.mps.is_available():
    dstr = 'mps'
device = torch.device(dstr)
print(dstr)

# Define the models (deterministic models) for the DDPG agent using mixin
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 10)
        self.linear_layer_2 = nn.Linear(10, 10)
        self.action_layer = nn.Linear(10, self.num_actions)

    def compute(self, inputs, role):
        states = inputs["states"].to(device)
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), {}


class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(
            self.num_observations + self.num_actions, 10)
        self.linear_layer_2 = nn.Linear(10, 10)
        self.linear_layer_3 = nn.Linear(10, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(
            torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


# Load and wrap the Gymnasium environment.
# Note: the environment version may change depending on the gymnasium version
#env = gym.make('MountainCarContinuous-v0')
env = gym.make("CartPole-v1")
env = MITM_trainer_env.PIDEnvTrainer(env)
env = wrap_env(env)
#env.to(device)


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs,
                      device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(
    env.observation_space, env.action_space, device)
models_ddpg["target_policy"] = DeterministicActor(
    env.observation_space, env.action_space, device)
models_ddpg["critic"] = DeterministicCritic(
    env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = DeterministicCritic(
    env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["random_timesteps"] = 100
cfg_ddpg["learning_starts"] = 0
cfg_ddpg["experiment"]["experiment_name"] = 'reward_max_at_one'
# logging to TensorBoard and write checkpoints each 300 and 1500 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 300
cfg_ddpg["experiment"]["checkpoint_interval"] = 1500


# Configure and instantiate the RL trainer
def train(name=None, pretrain=None):
    if name == None:
        name='demo'
    cfg_ddpg["experiment"]["experiment_name"] = name
    # logging to TensorBoard and write checkpoints each 300 and 1500 timesteps respectively
    cfg_ddpg["experiment"]["write_interval"] = 300
    cfg_ddpg["experiment"]["checkpoint_interval"] = 1500
    cfg_trainer = {"timesteps": 50000, "headless": True}
    agent_ddpg = DDPG(models=models_ddpg,
                      memory=memory,
                      cfg=cfg_ddpg,
                      observation_space=env.observation_space,
                      action_space=env.action_space,
                      device=device)
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)
    if pretrain:
        agent_ddpg.load(pretrain)

    print(cfg_ddpg)
    # start training
    trainer.train()

# load checkpoint


def load(file='./successful_models/DDPG_max_1.pt'):
    cfg_ddpg["experiment"]["experiment_name"] = 'demo animation'
    # logging to TensorBoard and write checkpoints each 300 and 1500 timesteps respectively
    cfg_ddpg["experiment"]["write_interval"] = 300
    cfg_ddpg["experiment"]["checkpoint_interval"] = 1500

    agent_ddpg = DDPG(models=models_ddpg,
                      memory=memory,
                      cfg=cfg_ddpg,
                      observation_space=env.observation_space,
                      action_space=env.action_space,
                      device=device)
    agent_ddpg.load(file)
    return agent_ddpg

def sim_and_render(agent_path='./runs/r_mov_slow/checkpoints/best_agent.pt', save_path=None):
    r = 'human'
    if save_path:
        frames = []
        states = []
        goals = []
        r = 'rgb_array'
    with torch.no_grad():
        env = gym.make("CartPole-v1", render_mode=r)
        env = MITM_env.MITMEnv(env, agent_path)
        env = PID_cartpole.PIDEnv(env)
        env = wrap_env(env)
        state, _ = env.reset()

        for i in range(500):
            if i < 10:
                step = 0
            elif i < 250:
                step = 1
            elif i < 500:
                step = 0
            state, reward, terminated, truncated, _ = env.step(torch.tensor(step,))
            if save_path:
                frames.append(env.render())
                states.append(state.detach().flatten().numpy())
                goals.append(step)
            if terminated:
                print('fell')
                state = env.reset()
        if save_path:
            PID_cartpole.save_frames_as_gif(frames, states, filename='MITM_env_push_multiple.gif', goals=goals)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('-b', '--bootstrap', type=str)
    parser.add_argument('-r', '--render', action="store_true")
    args = parser.parse_args()

    if args.render:
        exp = args.experiment
        if exp.endswith('.pt'):
            agent_path = exp
        else:
            agent_path = f'./runs/{args.experiment}/checkpoints/best_agent.pt'
        sim_and_render(agent_path)
    else:
        if args.bootstrap.endswith('.pt'):
            agent_path = args.bootstrap
        else:
            agent_path = f'./runs/{args.bootstrap}/checkpoints/best_agent.pt'
        train(args.experiment, agent_path)
