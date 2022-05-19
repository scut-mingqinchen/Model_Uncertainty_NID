clear all;

%% Gaussian Noise and Poisson Noise
model_path = 'results/exp_gaussian/gaussian_0.05';
%model_path = 'results/exp_poisson/peak_512';
psnr_list = zeros(8,7);
for n_ker = 1 : 8
    load(sprintf('testsets/Kernels/GT/Levin/%d.mat',n_ker));
    h_pad = floor(size(ker,1) / 2);
    w_pad = floor(size(ker,2) / 2);
    for n_im = 1 : 7
        ref = im2double(imread(sprintf('testsets/Images/GT/Set12/%02d.png',n_im)));
        im_name = sprintf('%s/kernel_%d/Set12/1-7/%02d/final.png',model_path,n_ker,n_im);
        if isfile(im_name)
            im = im2double(imread(im_name));
            psnr_list(n_ker,n_im) = psnr(crop(ref,h_pad,w_pad),crop(im,h_pad,w_pad));
        end
    end
end
mean(psnr_list(:))

%% Anger et al's datasets

% model_path = 'results/exp_anger';
% psnr_list = zeros(1,8);
% for n_im = 1 : 8
%     ker = imread(sprintf('testsets/Kernels/GT/Anger/%d.tif',n_im));
%     h_pad = floor(size(ker,1) / 2);
%     w_pad = floor(size(ker,2) / 2);
%     ref = im2double(imread(sprintf('testsets/Images/GT/Anger/%d.png',n_im)));
%     im_name = sprintf('%s/%d/final.png',model_path,n_im);
%     if isfile(im_name)
%         im = im2double(imread(im_name));
%         psnr_list(1,n_im) = psnr(crop(ref,h_pad,w_pad),crop(im,h_pad*2,w_pad*2));
%     end
% end
% mean(psnr_list(:))




function x = crop(x,h_pad,w_pad)
    if ismatrix(x)
        x = x(1+h_pad:end-h_pad,1+w_pad:end-w_pad);
    else
        x = x(1+h_pad:end-h_pad,1+w_pad:end-w_pad,:);
    end
end
