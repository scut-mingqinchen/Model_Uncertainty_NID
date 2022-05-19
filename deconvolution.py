import torch
from torch.optim import Adam
from model.unet import UNet
from model.pixel_masker import PixelMasker
from datetime import datetime
from tqdm import tqdm
from utils.degradtion_utils import torch_Ax
from loss import mask_loss
from utils.img_utils import save_imgs
from utils.common_utils import save_model, makedirs


def deconvolution(ys, pad_ys, k, sigmas, args):
    makedirs(args.img_dirs)
    args.h_pad, args.w_pad = k.shape[2] // 2, k.shape[3] // 2
    c, args.h, args.w = pad_ys.shape[1:4]
    args.c = c // args.n_file
    model = UNet(args).cuda()
    pixel_masker = PixelMasker(args)
    optimizer = Adam(model.parameters(), lr=args.lr)
    eta = (sigmas + args.eta_stabilizer) ** 2
    for iter in tqdm(range(args.n_iteration)):
        optimizer.zero_grad()
        sampled_ys, mask = pixel_masker(pad_ys)
        xs = model(sampled_ys)
        Kxs = torch_Ax(xs, k, args) ** (1. / args.gamma)
        loss = mask_loss(Kxs, ys, mask)
        mean_loss = torch.sum(torch.max(loss, eta))
        mean_loss.backward()
        optimizer.step()
        if (iter + 1) % args.n_save_freq == 0 or torch.max(loss - eta).item() < 0:
            print("loss in ", iter + 1, ":", mean_loss.item(), ' ', datetime.now().strftime("%H:%M:%S"))
            with torch.no_grad():
                avg = torch.zeros_like(pad_ys)
                for j in range(args.n_inference):
                    sampled_ys = pixel_masker(pad_ys)[0]
                    xs = model(sampled_ys)
                    avg += xs ** (1. / args.gamma)
                avg /= args.n_inference
                save_imgs(avg, args, postfix='%d' % (iter + 1))
                if torch.max(loss - eta).item() < 0 or (iter + 1) == args.n_iteration:
                    save_imgs(avg, args, postfix='final')
                    save_model(args, iter, model, optimizer, mean_loss)
                    break
