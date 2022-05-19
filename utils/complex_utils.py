import torch


def cdiv(x, y):
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c ** 2 + d ** 2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def csum(x, y):
    real = x[..., 0] + y
    img = x[..., 1]
    return torch.stack([real, img.expand_as(real)], -1)


def cabs2(x):
    return x[..., 0] ** 2 + x[..., 1] ** 2


def float2complex(x):
    return torch.stack([x, torch.zeros_like(x)], dim=-1)


def real(x):
    return x[..., 0]


def image(x):
    return x[..., 1]


def cmul(t1, t2):
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack(
        [real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def euler(theta):
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def angle(x):
    return torch.atan2(x[..., 1], x[..., 0])


def amp(x):
    return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
