import numpy as np
import cv2
import scipy.io as sio
from scipy.signal import fftconvolve
import torch.nn.functional as F
import torch


def load_ker(path):
    if path.find('.mat') != -1:
        ker_file = sio.loadmat(path)
        ker = np.array(ker_file['ker'], dtype=np.float32)
    elif path.find('.png') != -1 or path.find('.tif') != -1:
        ker = np.array(cv2.imread(path, -1), dtype=np.float32)
    else:
        raise NotImplementedError
    if ker.ndim == 3:
        ker = np.sum(ker, axis=2)
    ker = ker / np.sum(ker)
    return ker


def pad_for_kernel(img, kernel, mode):
    hy = (kernel.shape[0] - 1) // 2
    hx = (kernel.shape[0] - 1) - hy
    wy = (kernel.shape[1] - 1) // 2
    wx = (kernel.shape[1] - 1) - wy
    padding = [[hx, hy], [wx, wy]]
    return np.pad(img, padding, mode)


def edgetaper(img, kernel, n_tapers=3):
    '''tap edges for immitation of circulant boundary. '''
    alpha = edgetaper_alpha(kernel, img.shape)
    _kernel = kernel
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha * img + (1 - alpha) * blurred
    return img


def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1 - i), img_shape[i] - 1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def channel_with_edge_taper(img, kernel):
    if img.ndim == 2:
        pad_img = edgetaper(pad_for_kernel(img, kernel, 'edge'), kernel)
    else:
        pad_img = []
        for n_c in range(img.shape[2]):
            pad_img.append(edgetaper(pad_for_kernel(img[:, :, n_c], kernel, 'edge'), kernel))
        pad_img = np.stack(pad_img, axis=-1)
    return pad_img


def roll(psf, kernel_size, reverse=False):
    for axis, axis_size in zip([-2, -1], kernel_size):
        psf = torch.roll(psf,
                         int(axis_size / 2) * (-1 if not reverse else 1),
                         dims=axis)
    return psf


def p2o(psf, shape, onesided=False):
    kernel_size = (psf.size(-2), psf.size(-1))
    psf = F.pad(psf,
                [0, shape[1] - kernel_size[1], 0, shape[0] - kernel_size[0]])

    psf = roll(psf, kernel_size)
    psf = torch.rfft(psf, 2, onesided=onesided)
    return psf


def o2p(otf, kernel_size, onesided=False):
    psf = torch.irfft(otf, 2, onesided=onesided)
    psf = roll(psf, kernel_size, reverse=True)
    psf = psf[..., :kernel_size[-2], :kernel_size[-1]]
    return psf
