import torch


def heading_decorator(bottom=False, top=False, print_req=False):
    deco = '\n-------------------------------------------'
    if top is True:
        if print_req is False:
            return deco + '\n'
        else:
            print(deco + '\n')
    if bottom is True:
        if print_req is False:
            return deco
        else:
            print(deco)


def print_heading(heading, ):
    print(heading_decorator(top=True) + heading + heading_decorator(bottom=True))


def gym_torchify(gym_out, is_goal_env=False):
    observation, reward, done, info = gym_out
    if is_goal_env:
        new_observation = {}
        for k, v in observation.items():
            new_observation[k] = torch.Tensor(observation[k])
        return new_observation, torch.Tensor([reward]), torch.Tensor([done]), info
    else:
        return torch.Tensor(observation), torch.Tensor([reward]), torch.Tensor([done]), info

def ld_to_dl(batch_list_of_dicts):
    batch_dict_of_lists = {}
    for k in batch_list_of_dicts[0].keys():
        batch_dict_of_lists[k] = []
    for dictionary in batch_list_of_dicts:
        for k in batch_list_of_dicts[0].keys():
            batch_dict_of_lists[k].append(dictionary[k])
    return batch_dict_of_lists