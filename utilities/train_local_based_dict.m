function dicts = train_local_based_dict(Xh_patches, Xl_patches,dict_size)
%train_local_based_dict 此处显示有关此函数的摘要
%   此处显示详细说明



%%  PCA dimensionality reduction for features
% C = double(Xl_patches * Xl_patches');
% [V, D] = eig(C);
% D = diag(D); % perform PCA on features matrix
% D = cumsum(D) / sum(D);
% k = find(D >= 1e-3, 1); % ignore 0.1% energy
% V_pca = V(:, k:end); % choose the largest eigenvectors' projection
% Xl_patches_pca = V_pca' * Xl_patches;
% clear C D V

Xl_patches_pca =Xl_patches;

%%  train dicts using K-SVD
%  Set KSVD configuration
ksvd_conf.data = double(Xl_patches_pca);
clear Xl_patches_pca
ksvd_conf.iternum = 20; 
ksvd_conf.memusage = 'normal'; % higher usage doesn't fit...
ksvd_conf.dictsize = dict_size;
ksvd_conf.Tdata = 3; % maximal sparsity: TBD
ksvd_conf.samples = size(Xh_patches,2);

%  Training lr_dict and compute hr_dict
tic;
% fprintf('Training [%d x %d] dictionary on %d vectors using K-SVD\n', ...
%     size(ksvd_conf.data, 1), ksvd_conf.dictsize, size(ksvd_conf.data, 2))
[dl, gamma] = ksvd(ksvd_conf);
toc;


fprintf('Computing high-res. dictionary from low-res. dictionary\n');
Xh_patches = double(Xh_patches); % Since it is saved in single-precision.
dh = (Xh_patches * gamma')/(full(gamma * gamma'));

%%
dicts.dl = dl;
dicts.dh = dh;
% dicts.V_pca = V_pca;



end

