import warnings
from config import Config
import numpy as np
from utils.common_utils import save_codes, np2torch
from utils.ker_utils import load_ker
from utils.img_utils import load_img
from utils.degradtion_utils import np_y
from deconvolution import deconvolution

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model_name = 'exp_anger/'
    args = Config(model_name, gamma=2.2, nn_last_act='ReLU_Threshold')
    save_codes(root='.', save_path=args.model_dir + '/backup_code', except_path=["./results"])
    for n_im in range(8):
        ker_path = './testsets/Kernels/GT/Anger/%d.tif' % (n_im+1)
        file_list = ['./testsets/Images/Blurry/Anger/%d.tif' % (n_im + 1)]
        args.n_file = len(file_list)
        ker = load_ker(ker_path)
        ys, pad_ys, sigmas, dirs = [], [], [], []
        for i in range(args.n_file):
            y, name = load_img(file_list[i])
            y, pad_y, sigma = np_y(y, ker, args)
            ys.append(y), pad_ys.append(pad_y), sigmas.append(sigma), dirs.append(args.model_dir + name + '/')
        ys = np2torch(np.concatenate(ys, axis=1), is_cuda=True)
        pad_ys = np2torch(np.concatenate(pad_ys, axis=1), is_cuda=True)
        ker = np2torch(ker[np.newaxis, np.newaxis, :, :], is_cuda=True)
        sigmas = np2torch(np.array(sigmas, dtype=np.float32), is_cuda=True)
        args.img_dirs = dirs
        deconvolution(ys, pad_ys, ker, sigmas, args)
