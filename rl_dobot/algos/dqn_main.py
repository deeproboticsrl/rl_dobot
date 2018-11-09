import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
import os
import numpy as np
from rl_dobot.algos.learn import DQN_LEARN
from dobot_gym.envs import LineReachEnv
from rl_dobot.utils import Buffer, gym_torchify, ld_to_dl
# import ipdb
parser = argparse.ArgumentParser(description='DQN dobot')
parser.add_argument('--env_seed', type=int, default=1105, help="environment seed")
parser.add_argument('--target_update_interval', type=int, default=1000, help="update target network")
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma')
parser.add_argument('--save_iter', type=int, default=20, help='save model and buffer ')
parser.add_argument('--reward_scale', type=float, default=0.3, help='scale reward')
parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
parser.add_argument('--sample_batch_size', default=500, type=int, help="number of samples in a batch")
parser.add_argument('--epsilon_max_steps', default=500000, type=int,help = "total number of epsilon steps ")
parser.add_argument('--buffer_capacity',default=1000000,type=int , help ="capacity of buffer")

args = parser.parse_args()
logdir = "./tensorboard_log/"
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(log_dir=logdir)

env = LineReachEnv()
print('env running')
# ipdb.set_trace()
action_dim = (env.action_space.shape)
# state_dim = env.observation_space.shape  ## CHECK
env.seed(args.env_seed)
torch.manual_seed(args.env_seed)
np.random.seed(args.env_seed)

buffer = Buffer(capacity=args.buffer_capacity)

epsilon = 0.5
epsilon_max = 1.0
epsilon_min = 0.1
episolon_max_steps = args.epsilon_max_steps


def epsilon_greedy_action(action, step):
    epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * step / episolon_max_steps)
    if np.random.rand() < epsilon:
        return env.action_space.sample() - 1
    return action

dqn = DQN_LEARN(writer=writer, reward_scale=args.reward_scale, discount_factor=args.discount_factor,
                action_dim=action_dim, target_update_interval=args.target_update_interval,
                lr=args.lr)


step = 1  # CAUTION: step should not be zero
max_reward = -np.inf
rewards = []
for i in range(args.epochs):
    episode_reward = 0

    state = torch.Tensor(env.reset())
    done = False
    timestep = 0
    while (not done or timestep <= args.max_timestep):
        policyaction = dqn.get_action(state)

        action = epsilon_greedy_action(policyaction, step)
        observation, reward, done = gym_torchify(env.step(action))
        sample = dict(state=state, action=action, reward=reward, next_state=observation, done=done)
        buffer.add(sample)

        if len(buffer) > 5 * args.sample_batch_size:
            batch_list_of_dicts = buffer.sample(batch_size=args.sample_batch_size)
            batch_dict_of_lists = ld_to_dl(batch_list_of_dicts)

            ## Combined experience Replay

            for k in batch_list_of_dicts[0].keys():
                batch_dict_of_lists[k].append(sample[k])
            dqn.policy_update(batch_dict_of_lists, update_number=step)
            step += 1

        episode_reward += reward
        state = observation
        timestep += 1

    if episode_reward > max_reward:
        max_reward = episode_reward
        # save current best model
        print(f"\nNew best model with reward {max_reward}")
        dqn.save_model(env_name=args.env_name, info='best')

    if i % args.save_iter == 0:
        print(f"\nSaving periodically - iteration {i}")
        dqn.save_model(env_name=args.env_name, info=str(i))
        buffer.save_buffer(info=args.env_name)

    dqn.writer.add_scalar("Episode Reward", episode_reward, i)
    rewards.append(episode_reward)
