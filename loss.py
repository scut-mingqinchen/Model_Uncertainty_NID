import torch


def mask_loss(output, target, mask):
    loss = torch.sum((1 - mask) * (output - target) ** 2, dim=[0, 2, 3]) / torch.sum(1 - mask, dim=[0, 2, 3])
    return loss
