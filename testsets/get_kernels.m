for i = 9:14
    ker = eval(sprintf('kernel%d',i));
    save(sprintf('Kernels/GT/IDBM3D/%d.mat',i-8),'ker');
end