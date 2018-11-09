'''
update rule for the DQN model is defined
'''

'''
Preprocessing : give only hsv value or masked end efforctor image
Downsample pixel and convert RGB to grayscale
Use nn.functional where there is no trainable parameter like relu or maxpool
Try pooling vs not pooling
In Atari they consider multiple frames to find direction
'''

'''
To reduce the size of the representation 
using larger stride in CONV layer once in a while can always be a preferred option in many cases. 
Discarding pooling layers has also been found to be important in training good generative models, 
such as variational autoencoders (VAEs) or generative adversarial networks (GANs).
Also it seems likely that future architectures will feature very few to no pooling layers.
Hintons says pooling is a mistake and that it works so well is a disaster

Pooling is important when we don't consider the position of the object in the image , just its presence
'''
import os
import torch
import torch.nn as nn
import torch.functional as F
from rl_dobot.utils import Buffer
from rl_dobot.utils import hard_update, print_heading, heading_decorator
from rl_dobot.algos.dqn_model import DQN


def to_np(x):
    return x.data.cpu().numpy()


device = torch.device("cpu")


class DQN_LEARN():
    def __init__(self, reward_scale, discount_factor, action_dim, target_update_interval, lr, writer):
        self.Q_net = DQN(num_action=action_dim)
        self.target_Q_net = DQN(num_action=action_dim)

        self.reward_scale = reward_scale
        self.discount_factor = discount_factor
        self.action_dim = action_dim

        self.writer = writer

        self.target_update_interval = target_update_interval
        self.lr = lr
        hard_update(self.Q_net, self.target_Q_net)
        # self.target_Q_net.load_state_dict(self.Q_net)  this can also be done
        self.optimizer = torch.optim.Adam(lr=self.lr)

    def policy_update(self, batch, update_number):

        state_batch = torch.stack(batch['state']).detach()
        action_batch = torch.stack(batch['action']).detach()
        reward_batch = torch.stack(batch['reward']).detach()
        next_state_batch = torch.stack(batch['next_state']).detach()
        done_batch = torch.stack(batch['done']).detach()

        q_values = self.Q_net(state_batch)
        # next_q_values = self.Q_net(next_state_batch)

        q_t1 = self.target_Q_net(next_state_batch).detach()

        # target = reward + gamma * max(Q'(s',a')
        # loss = (r+ gamma*max(Q'(s',a') - Q(s,a))^2
        # reward clipping
        Q_target = reward_batch * self.reward_scale + self.gamma * (1 - done_batch) * torch.max(q_t1, dim=1).view(-1, 1)
        ## Huber loss
        huber_loss = nn.SmoothL1Loss()

        error = huber_loss(Q_target - q_values)

        ## backward pass
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

        if update_number % self.target_update_interval == 0:
            hard_update(self.Q_net, self.target_Q_net)

        self.writer.add_scaler("q_values", q_values.mean(), global_step=update_number)
        self.writer.add_scaler("qt1_values", q_t1.mean(), global_step=update_number)
        self.writer.add_scaler("huber_loss", huber_loss.mean(), global_step=update_number)
        self.writer.add_scaler("error", error.mean(), global_step=update_number)

    def save_model(self, env_name, q_path, info=1):
        if q_path is not None:
            self.q_path = q_path
        else:
            self.q_path = f'model/{env_name}/'

        os.makedirs(self.q_path, exist_ok=True)

        print_heading("Saving actor,critic,value network parameters")
        torch.save(self.target_Q_net.state_dict(), q_path + f"value_{info}.pt")
        heading_decorator(bottom=True, print_req=True)

    def load_model(self, q_path=None):
        print_heading(f"Loading models from paths: \n q_func:{q_path}")
        if q_path is not None:
            self.target_Q_net.load_state_dict(torch.load(q_path))

        print_heading('loading done')

    def get_action(self, state):
        return self.target_Q_net.get_action(state)
