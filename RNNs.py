import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


def init_models(env, device):
    models_ddpg = {}
    models_ddpg["policy"] = Actor(
        env.observation_space, env.action_space, device, num_envs=env.num_envs
    )
    models_ddpg["target_policy"] = Actor(
        env.observation_space, env.action_space, device, num_envs=env.num_envs
    )
    models_ddpg["critic"] = Critic(
        env.observation_space, env.action_space, device, num_envs=env.num_envs
    )
    models_ddpg["target_critic"] = Critic(
        env.observation_space, env.action_space, device, num_envs=env.num_envs
    )
    return models_ddpg


class Actor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        num_envs=1,
        num_layers=1,
        hidden_size=400,
        sequence_length=20,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(
            input_size=self.num_observations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # batch_first -> (batch, sequence, features)

        self.linear_layer_1 = nn.Linear(self.hidden_size, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"].to(self.device)
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = states.view(
                -1, self.sequence_length, states.shape[-1]
            )  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            sequence_index = (
                1 if role == "target_policy" else 0
            )  # target networks act on the next state of the environment
            hidden_states = hidden_states[
                :, :, sequence_index, :
            ].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (
                        terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1
                    ).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(
                        rnn_input[:, i0:i1, :], hidden_states
                    )
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(
                -1, 1, states.shape[-1]
            )  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(
            rnn_output, start_dim=0, end_dim=1
        )  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = F.relu(self.linear_layer_1(rnn_output))
        x = F.relu(self.linear_layer_2(x))

        # Pendulum-v1 action_space is -2 to 2
        return 0.5 * torch.tanh(self.action_layer(x)), {"rnn": [hidden_states]}


class Critic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        num_envs=1,
        num_layers=1,
        hidden_size=400,
        sequence_length=20,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(
            input_size=self.num_observations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # batch_first -> (batch, sequence, features)

        self.linear_layer_1 = nn.Linear(self.hidden_size + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # critic is only used during training
        rnn_input = states.view(
            -1, self.sequence_length, states.shape[-1]
        )  # (N, L, Hin): N=batch_size, L=sequence_length

        hidden_states = hidden_states.view(
            self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
        )  # (D * num_layers, N, L, Hout)
        # get the hidden states corresponding to the initial sequence
        sequence_index = (
            1 if role == "target_critic" else 0
        )  # target networks act on the next state of the environment
        hidden_states = hidden_states[
            :, :, sequence_index, :
        ].contiguous()  # (D * num_layers, N, Hout)

        # reset the RNN state in the middle of a sequence
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            indexes = (
                [0]
                + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                + [self.sequence_length]
            )

            for i in range(len(indexes) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                rnn_output, hidden_states = self.rnn(
                    rnn_input[:, i0:i1, :], hidden_states
                )
                hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                rnn_outputs.append(rnn_output)

            rnn_output = torch.cat(rnn_outputs, dim=1)
        # no need to reset the RNN state in the sequence
        else:
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(
            rnn_output, start_dim=0, end_dim=1
        )  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = F.relu(
            self.linear_layer_1(torch.cat([rnn_output, inputs["taken_actions"]], dim=1))
        )
        x = F.relu(self.linear_layer_2(x))

        return self.linear_layer_3(x), {"rnn": [hidden_states]}
