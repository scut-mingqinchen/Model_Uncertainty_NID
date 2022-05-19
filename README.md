# Nonblind Image Deconvolution via Leveraging Model Uncertainty in An Untrained Deep Neural Network
In this repository we provide the official implementation of "Nonblind Image Deconvolution via Leveraging Model
Uncertainty".
## General Information
- Codename: Model Uncertainty NID (IJCV 2022)
- Writers: Mingqin Chen (csmingqinchen@mail.scut.edu.cn); Yuhui Quan (csyhquan@scut.edu.cn); Tongyao Pang (matpt@nus.edu.sg); Hui
  Ji (matjh@nus.edu.sg)
- Institute: School of Computer Science and Engineering, South China
  University of Technology; Department of Mathematics, National University of Singapore

For more information please see:
- [[paper]](https://link.springer.com/article/10.1007/s11263-022-01621-9)
- [[supmat]](https://static-content.springer.com/esm/art%3A10.1007%2Fs11263-022-01621-9/MediaObjects/11263_2022_1621_MOESM1_ESM.pdf)
- [[website]](https://csyhquan.github.io/)


## Requirements
Here is the list of libraries you need to install to execute the code:
* Python 3.6
* PyTorch 1.6.0
* scikit-image
* scipy
* cv2 (opencv for python)

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install scikit-image
```

## How to Execute Demo
1. Run the demo code in `demo_gaussian_noise.py` for NID in the presence of AWGN.
2. Run the demo code in `demo_poisson_noise.py` for NID in the presence of Poisson noise.
3. Run the demo code in `demo_real_world.py` for NID in Real-World Cases.
4. Run the demo code in `demo_kohler.py` for NID with inaccurate kernels.
5. Run `get_psnr_ssim_list.m` in Matlab for quantitative results.   

## Citation
```
@Article{chen2022nonblind,
  author    = {Chen, Mingqin and Quan, Yuhui and Pang, Tongyao and Ji, Hui},
  title     = {Nonblind Image Deconvolution via Leveraging Model Uncertainty in An Untrained Deep Neural Network},
  journal   = {International Journal of Computer Vision},
  year      = {2020},
  publisher = {Springer}
}
```

## Contacts
For questions, please send an email to **csmingqinchen@mail.scut.edu.cn**
