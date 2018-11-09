import torch.nn as nn


# xavier initialization does random operation so calling it twice on same network will yield different results
def xavier_initialisation(*module):
    modules = [*module]
    for i in range(len(modules)):
        if type(modules[i]) in [nn.Linear]:
            nn.init.xavier_normal_(modules[i].weight.data)


def no_activation(x):
    return x


def soft_update(original, target, t=0.005):
    # zip(a,b) is same as [a.b]
    for original_param, target_param in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data * t + target_param * (1 - t))
        # check copy_ parameter : Something on cpu or gpu, also no need for return as it changes in self


def hard_update(original, target):
    for original_param, target_param in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data)


def None_grad(optimiser):
    for group in optimiser.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.detach_()
                p.grad = None
