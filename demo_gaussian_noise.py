import warnings
from config import Config
import numpy as np
from utils.common_utils import save_codes, np2torch
from utils.ker_utils import load_ker
from utils.img_utils import load_img
from utils.degradtion_utils import np_Ax
from deconvolution import deconvolution

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    for sigma in [0.05, 0.03, 0.01]:
        for n_ker in range(1, 9):
            model_name = 'exp_gaussian/gaussian_%.2f/kernel_%d/Set12/1-7' % (sigma, n_ker)
            args = Config(model_name, is_gaussian=True, sigma=sigma)
            save_codes(root='.', save_path=args.model_dir + '/backup_code', except_path=["./results"])
            ker_path = './testsets/Kernels/GT/Levin/%d.mat' % n_ker
            file_list = ['./testsets/Images/GT/Set12/%02d.png' % (i + 1) for i in range(7)]
            args.n_file = len(file_list)
            ker = load_ker(ker_path)
            ys, pad_ys, sigmas, dirs = [], [], [], []
            for i in range(args.n_file):
                x, name = load_img(file_list[i])
                y, pad_y, sigma = np_Ax(x, ker, args)
                ys.append(y), pad_ys.append(pad_y), sigmas.append(sigma), dirs.append(args.model_dir + name + '/')
            ys = np2torch(np.concatenate(ys, axis=1), is_cuda=True)
            pad_ys = np2torch(np.concatenate(pad_ys, axis=1), is_cuda=True)
            ker = np2torch(ker[np.newaxis, np.newaxis, :, :], is_cuda=True)
            sigmas = np2torch(np.array(sigmas, dtype=np.float32), is_cuda=True)
            args.img_dirs = dirs
            deconvolution(ys, pad_ys, ker, sigmas, args)
