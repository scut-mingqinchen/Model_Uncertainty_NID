import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve
import numpy as np
import noise_est.noise_estimation as ns
from scipy.signal import fftconvolve
from utils.ker_utils import channel_with_edge_taper, p2o
from utils.complex_utils import angle, cmul, amp


def torch_Ax(x, k, args):
    if args.degradtion_type == 'conv':
        k = torch.rot90(k, 2, dims=[-1, -2])
        y = F.conv2d(x.permute(1, 0, 2, 3), k)
        y = y.permute(1, 0, 2, 3)
    elif args.degradtion_type == 'fft_conv':
        Fk = p2o(k, x.shape[-2:])
        Fx = torch.rfft(x, signal_ndim=2, onesided=False)
        Fy = cmul(Fx, Fk)
        y = torch.irfft(Fy, signal_ndim=2, onesided=False)
        y = y[:, :, args.h_pad:-args.h_pad, args.w_pad:-args.w_pad]
    elif args.degradtion_type == 'fft_conv_with_perturbation':
        noise = torch.randn_like(k) * 0.001
        Fk = torch.rfft(k, signal_ndim=2, onesided=False)
        phase_k = angle(Fk)
        Fn = torch.rfft(noise, signal_ndim=2, onesided=False)
        norm_n = amp(Fn)
        Fn = torch.stack([torch.cos(phase_k) * norm_n, torch.sin(phase_k) * norm_n], dim=-1)
        n = torch.irfft(Fn, signal_ndim=2, onesided=False)
        kn = k + n
        kn = kn / torch.sum(kn)
        temp_args = args
        temp_args.degradtion_type = 'fft_conv'
        y = torch_Ax(x, kn, temp_args)
    else:
        raise NotImplementedError
    return y


def np_Ax(x, k, args):
    y = fftconvolve(x, k, 'valid')
    if args.is_gaussian:
        y = y + np.random.normal(size=np.shape(y)) * args.sigma
    if args.is_poisson:
        y = np.random.poisson(y * args.peak) / args.peak
    return np_y(y, k, args)


def np_y(y, k, args):
    if args.is_noise_blind:
        sigma = ns.noise_estimate(y, 8)
    else:
        sigma = args.sigma
    pad_y = channel_with_edge_taper(y, k)
    if y.ndim == 2:
        y = y[np.newaxis, :, :, np.newaxis]
        pad_y = pad_y[np.newaxis, :, :, np.newaxis]
    else:
        y = y[np.newaxis, :, :, :]
        pad_y = pad_y[np.newaxis, :, :, :]
    y = np.transpose(y, [0, 3, 1, 2])
    pad_y = np.transpose(pad_y, [0, 3, 1, 2])
    return y, pad_y, sigma
