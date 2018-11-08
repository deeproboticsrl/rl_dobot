## DQN implementation
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.module):
    def __init__(self,num_action = 4):
        # batch x  channel x height x width
        self.conv1 = nn.Conv2d(3,20,(5,5),stride=(2,2))
        # self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20,40,(5,5),stride=(2,2))
        # self.bn2 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(40 * 5 * 5,500)
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,out_features=num_actions)
    def forward(self,x):
        x = F.max_pool2d(F.relu((self.conv1(x))), (4,4))
        x = F.max_pool2d(F.relu((self.conv2(x))), (4,4))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        #x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    '''
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    '''



