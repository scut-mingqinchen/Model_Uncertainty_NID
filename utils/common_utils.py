import os
from shutil import copyfile
import torch
import numpy as np


def torch2np(x_tensor):
    if isinstance(x_tensor, np.ndarray):
        return x_tensor
    elif not x_tensor.is_cuda:
        x = x_tensor.numpy()
        return x
    else:
        x = x_tensor.detach().cpu().numpy()
        return x


def np2torch(x, is_cuda=False):
    if isinstance(x, torch.Tensor):
        pass
    else:
        x = torch.from_numpy(x.copy())
        x = x.type(torch.float32)
    if is_cuda:
        x = x.cuda()
    return x


def batch_np_func(func, imgs, **kwargs):
    res_type = imgs.type()
    imgs = imgs.cpu().numpy()
    results = []
    for i in range(imgs.shape[0]):
        results.append(torch.from_numpy(func(imgs[i, 0], **kwargs)[np.newaxis, np.newaxis, :, :]))
    results = torch.cat(results, 0).type(res_type)
    return results


def batch_torch_func(func, imgs, **kwargs):
    results = []
    for i in range(imgs.shape[0]):
        results.append(func(imgs[i, 0], **kwargs).unsqueeze(0).unsqueeze(0))
    results = torch.cat(results, 0)
    return results


def save_codes(root='.', save_path='./results/0630/test_save_all_code', except_path=None):
    if except_path is None:
        except_path = ["./results", "./ref_code"]
    list = os.listdir(root)
    for i in list:
        file = root + '/' + i
        if file in except_path:
            continue
        if file.find('.py') == len(file) - 3 or file.find('.m') == len(file) - 2:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            copyfile(file, save_path + '/' + i)
        elif os.path.isdir(file):
            save_codes(file, save_path + '/' + i, except_path)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_model(args, iter, model, optimizer, loss):
    torch.save({
        'epoch': iter + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, args.model_save_dir + 'checkpoint.pth')


def makedirs(dir_list):
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)
