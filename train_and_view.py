#!python3

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
import os
import RNNs
import MLPs

# Import the skrl components to build the RL system
import skrl
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.envs.torch import wrap_env
from skrl.resources.noises.torch import GaussianNoise
from torch.optim.lr_scheduler import StepLR

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dstr = "cpu"
if torch.cuda.is_available():
    dstr = "cuda"
elif torch.backends.mps.is_available():
    dstr = "mps"
device = torch.device(dstr)
# device = torch.device('cpu')
print(dstr)


# Configure and instantiate the RL trainer
def train(name=None, pretrain=None):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = MITM_trainer_env.PIDEnvTrainer(env)
    env = wrap_env(env)
    print(env.observation_space)
    if name == None:
        name = "demo"
        # Instantiate a RandomMemory (without replacement) as experience replay memory
    memory = RandomMemory(
        memory_size=15000, num_envs=env.num_envs, device=device, replacement=False
    )
    cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
    cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(
        theta=0.15, sigma=0.1, base_scale=1.0, device=device
    )
    cfg_ddpg["discount_factor"] = 0.98
    cfg_ddpg["batch_size"] = 100
    cfg_ddpg["random_timesteps"] = 0
    cfg_ddpg["learning_starts"] = 1000
    cfg_ddpg["experiment"]["experiment_name"] = name
    cfg_ddpg["experiment"]["store_seperately"] = True
    # logging to TensorBoard and write checkpoints each 300 and 1500 timesteps respectively
    cfg_ddpg["experiment"]["write_interval"] = 300
    cfg_ddpg["experiment"]["checkpoint_interval"] = 1500

    # cfg_ddpg["learning_rate_scheduler"] = StepLR
    # cfg_ddpg["learning_rate_scheduler_kwargs"] = {"step_size": 1000, "gamma": 0.95}

    cfg_trainer = {"timesteps": 100000, "headless": True}

    if pretrain:
        models_ddpg = MLPs.load_models(
            pretrain, env.action_space, env.observation_space, device
        )
    else:
        models_ddpg = RNNs.init_models(env, device)

    agent_ddpg = DDPG(
        models=models_ddpg,
        memory=memory,
        cfg=cfg_ddpg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    # agent_ddpg.init()
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)

    print(cfg_ddpg)
    # start training
    trainer.train()


def load(action_space, observation_space, file="./successful_models/DDPG_max_1.pt"):
    # logging to TensorBoard and write checkpoints each 300 and 1500 timesteps respectively
    models_ddpg = MLPs.load_models(file, action_space, observation_space)
    cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
    cfg_ddpg["random_timesteps"] = 0
    cfg_ddpg["learning_starts"] = 10000
    cfg_ddpg["experiment"]["experiment_name"] = "demo animation"
    # Instantiate a RandomMemory (without replacement) as experience replay memory
    memory = RandomMemory(
        memory_size=15000, num_envs=1, device=device, replacement=False
    )
    agent_ddpg = DDPG(
        models=models_ddpg,
        memory=memory,
        cfg=cfg_ddpg,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )
    # agent_ddpg.load(file)
    agent_ddpg._exploration_noise = None
    agent_ddpg.init()
    return agent_ddpg


def sim_and_render(
    agent_path="./runs/r_mov_slow/checkpoints/best_agent.pt", save_path=None
):
    r = "human"
    if save_path:
        frames = []
        states = []
        goals = []
        attacks = []
        r = "rgb_array"
        os.makedirs(save_path)
    with torch.no_grad():
        env = gym.make("CartPole-v1", render_mode=r)
        env = MITM_env.MITMEnv(
            env, agent_path, noise_path="./successful_models/blind/unlimited_actions.pt"
        )
        env = PID_cartpole.PIDEnv(env)
        env = wrap_env(env)
        state, _ = env.reset()
        step = 0
        for i in range(10000):
            if i % 100 == 0:
                step = np.random.randint(-2, 3)

            state, reward, terminated, truncated, _ = env.step(
                torch.tensor(step, dtype=torch.int)
            )

            if save_path:
                frames.append(env.render())
                states.append(state.detach().flatten().numpy())
                goals.append(step)
            if terminated:
                print("fell")
                state = env.reset()
        if save_path:
            np.savetxt(
                f"{save_path}/states.csv",
                states,
                delimiter=",",
                header="Position , Speed, Angle, Angular speed, attack",
            )
            np.savetxt(f"{save_path}/goals.csv", goals, delimiter=",", header="Command")
            PID_cartpole.save_frames_as_gif(frames, states, path=save_path, goals=goals)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("-b", "--bootstrap", type=str)
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-s", "--savepath", type=str)
    args = parser.parse_args()
    boots = args.bootstrap
    if args.render:
        savepath = args.savepath if args.savepath else None
        exp = args.experiment
        if exp.endswith(".pt"):
            agent_path = exp
        else:
            agent_path = f"./runs/{exp}/checkpoints/best_agent.pt"
        sim_and_render(agent_path, savepath)
    elif boots:
        if boots.endswith(".pt"):
            agent_path = boots
        else:
            agent_path = f"./successful_models/{boots}.pt"
        train(args.experiment, agent_path)
    else:
        train(args.experiment)
