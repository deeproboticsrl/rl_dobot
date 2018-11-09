## DQN implementation


import torch.nn as nn
import torch.nn.functional as F
import torch

def to_np(x):
    return x.data.cpu().numpy()
class DQN(nn.Module):
    def __init__(self, num_action=3):
        super().__init__()
        self.num_action = num_action
        # batch x  channel x height x width
        self.conv1 = nn.Conv2d(3, 20, (5, 5), stride=(2, 2))
        # print(self.conv1)
        # self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 40, (5, 5), stride=(2, 2))
        # self.bn2 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(40 * 5 * 5, 200)
        self.fc2 = nn.Linear(204, 50)
        self.fc3 = nn.Linear(50, out_features=self.num_action)

    def forward(self, input):
        x = F.max_pool2d(F.relu((self.conv1(input["image"]))), (2, 2))
        x = F.max_pool2d(F.relu((self.conv2(x))), (2, 2))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # x.size(0) = batch size
        # x = x.view(-1, self.num_flat_features(x))
        y = torch.cat((x, input["pose"]), dim=1)
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y
    def get_action(self,state):
        q_values = self.forward(state)
        action = to_np(torch.argmax(q_values)) - 1
        return action
    '''  
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    '''


if __name__ == '__main__':
    ob = DQN()
    print (ob.state_dict())
