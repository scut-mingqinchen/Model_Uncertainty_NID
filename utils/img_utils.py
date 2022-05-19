from utils.common_utils import torch2np
import numpy as np
import cv2


def save_imgs(imgs, args, postfix=''):
    imgs = torch2np(imgs)
    imgs = np.concatenate(np.split(imgs, args.n_file, axis=1), axis=0)
    for i in range(args.n_file):
        save_img(imgs[i:i + 1], args.img_dirs[i] + postfix + '.png')


def save_img(img, save_name):
    img = torch2np(img)
    img = np.squeeze(np.transpose(img, [0, 2, 3, 1]))
    cv2.imwrite(save_name, np.uint8(np.clip(img, 0, 1) * 255. + 0.5))


def load_img(path):
    img_name = path[path.rfind('/'):path.rfind('.')]
    img = np.array(cv2.imread(path, -1), dtype=np.float32) / 255.
    return img, img_name


def cut_boundary(img, h_pad, w_pad):
    return img[:, :, h_pad:-h_pad, w_pad:-w_pad]
