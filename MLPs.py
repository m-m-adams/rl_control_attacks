import gymnasium as gym
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import skrl
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.envs.torch import wrap_env
from skrl.resources.noises.torch import GaussianNoise
from torch.optim.lr_scheduler import StepLR


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
        states = inputs["states"].to(self.device)
        x = F.relu(self.linear_layer_1(states))
        x = F.relu(self.linear_layer_2(x))
        # action space is 1 - -1, useful action space is smaller
        return 0.5 * torch.tanh(self.action_layer(x)), {}


class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 10)
        self.linear_layer_2 = nn.Linear(10, 10)
        self.linear_layer_3 = nn.Linear(10, 1)

    def compute(self, inputs, role):
        x = F.relu(
            self.linear_layer_1(
                torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
            )
        )
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


def init_models(env, device, pretrain_path=None):
    # Instantiate the agent's models (function approximators).
    # DDPG requires 4 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
    models_ddpg = {}
    models_ddpg["policy"] = DeterministicActor(
        env.observation_space, env.action_space, device
    )
    models_ddpg["target_policy"] = DeterministicActor(
        env.observation_space, env.action_space, device
    )
    models_ddpg["critic"] = DeterministicCritic(
        env.observation_space, env.action_space, device
    )
    models_ddpg["target_critic"] = DeterministicCritic(
        env.observation_space, env.action_space, device
    )

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_ddpg.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return models_ddpg


def load_models(path, action_space, observation_space, device):
    models_ddpg = {}
    models_ddpg["policy"] = DeterministicActor(observation_space, action_space, device)
    models_ddpg["target_policy"] = DeterministicActor(
        observation_space, action_space, device
    )
    models_ddpg["critic"] = DeterministicCritic(observation_space, action_space, device)
    models_ddpg["target_critic"] = DeterministicCritic(
        observation_space, action_space, device
    )

    old = DDPG(models=models_ddpg)
    old.load(path)
    return old.models
