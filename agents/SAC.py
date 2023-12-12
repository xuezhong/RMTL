import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .DDPG_ESMM import DDPG_wESMMAgent
import warnings

warnings.filterwarnings("ignore")


class SACAgent(DDPG_wESMMAgent):
    def __init__(self, env: gym.Env, actor_name, arguments):
        super_args = dict(
            env=env,
            actor_name=actor_name,
            embed_dim=arguments.embed_dim,
            ou_noise_theta=arguments.ou_noise_theta,
            ou_noise_gamma=arguments.ou_noise_gamma,
            gamma=arguments.gamma,
            memory_size=arguments.memory_size,
            batch_size=arguments.batch_size,
            drop_out=arguments.drop_out,
            pretrain_path=arguments.pretrain_path,
            actor_lr=arguments.actor_lr,
            critic_lr=arguments.critic_lr,
            actor_reg=arguments.actor_reg,
            tau=arguments.tau,
            soft_update_freq=arguments.soft_update_freq,
            actor_update_freq=arguments.actor_update_freq,
            init_training_step=arguments.init_training_step,
            ips=arguments.ips,
        )
        super(SACAgent, self).__init__(**super_args)
        self.actor_target_optimizer = optim.Adam(self.actor_target.parameters(), lr=5*self.actor_lr)


    def __str__(self):
        return "SAC"
