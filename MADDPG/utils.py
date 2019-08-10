import torch

"""
From Udacity
"""

def transpose_list(nd_list):
    return list(map(list,zip(*nd_list)))

def transpose_to_tensor(nd_list):
    to_tensor = lambda x: torch.tesnor(x, dtype=torch.float)
    return list(map(to_tensor,zip(*nd_list)))