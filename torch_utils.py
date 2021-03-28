import torch

def get_param_num(model):
    sum_param=0
    for param in model.parameters():
        sum_param += param.numel()
    return sum_param


