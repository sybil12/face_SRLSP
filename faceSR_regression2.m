
% =========================================================================
% local patch based global regression face super resolution
% version1.1 回归器学习在SR之前，使用row patch进行计算
% version2.0 学习局部字典，代价太高？待实现
%=========================================================================


% clc; close all;
clear  % clear all;
addpath('./utilities');

% set parameters
% nrow       = 120;             % rows of HR face image
% ncol        = 100;            % cols of HR face image
nTraining   = 360;              % number of training sample
nTesting    = 40;                   % number of test sample
upscale     = 2;                    % upscaling factor
patch_size  = 8;                    % image patch (window) size
overlap     = patch_size-2;         % the overlap between neighborhood patches
% dict_size = 6;                  % num of atoms in a dict
% clusterszA = 240;               % num of patches used for counting projection matrix

% construct the HR and LR training pairs from the FEI face database
str = ['./data/YH_YL_FEI' '_s' num2str(upscale) '.mat'];
if exist(str,'file')
    load(str,'YH','YL');
else
    [YH, YL] = Training_LH(upscale,nTraining); %double or im2double 类型
    save(str,'YH','YL');
end
[imrow, imcol, ~] = size(YH);


lambda = 0.01;
% load or compute GR regression
str = ['./data/YH_YL_FEI_dict-PP' '_s' num2str(upscale) '.mat'];
if exist(str,'file')
    disp('Load trained GR-PPs...');
    load(str);
else
    disp('Training GR-PPs...');
    tic
    [PPs,dicts] = trainP_faceSR_for_dict(YH, YL, imrow, imcol, upscale,patch_size,overlap, lambda);
    toc
    save(str,'PPs','dicts')
    disp('Done.');
end
% dict_str = ['./data/YH_YL_FEI_' num2str(U) 'x' num2str(V) '_on_' num2str(dict_size) 'atoms_dicts.mat' ];
% load(str,'dicts');



bb_psnr = zeros(1,nTesting);
bb_ssim = zeros(1,nTesting);
nn_psnr = zeros(1,nTesting);
nn_ssim = zeros(1,nTesting);
sr_psnr = zeros(1,nTesting);
sr_ssim = zeros(1,nTesting);
fprintf('\nface hallucinating _%dX for %d input test images\n', upscale, nTesting);

for TestImgIndex = 1:nTesting
    fprintf('\nProcessing  %d/%d LR image\n', TestImgIndex,nTesting);
    
    % read ground truth of one test face image
    strh = strcat('./testFaces/',num2str(TestImgIndex),'_test.jpg');
    im_h = imread(strh);
    im_h = double(im_h);
    
    % use nearest method to get  lr img
    im_n = imresize(im_h,1/upscale,'nearest');
    im_l = im_n;
    
    % face SR 
    tic
%     [im_SR] = faceSR(im_l,YH,YL,upscale,patch_size,overlap);
    [im_SR] = faceSR2(im_l,PPs,dicts,imrow, imcol, upscale,patch_size,overlap);
    toc
    
    % bicubic interpolation for reference
    im_b = imresize(im_l, [imrow,imcol], 'bicubic');
    % nearest interpolation for reference
    im_n = imresize(im_n,[imrow,imcol],'nearest');
    
    im_h = uint8(im_h);
    im_n = uint8(im_n);
    im_b = uint8(im_b);
    im_SR = uint8(im_SR);

    % compute PSNR and SSIM for Bicubic and our method
    nn_psnr(TestImgIndex) = psnr(im_n,im_h);
    nn_ssim(TestImgIndex) = ssim(im_n,im_h);
    bb_psnr(TestImgIndex) = psnr(im_b,im_h);
    bb_ssim(TestImgIndex) = ssim(im_b,im_h);
    sr_psnr(TestImgIndex) = psnr(im_SR,im_h);
    sr_ssim(TestImgIndex) = ssim(im_SR,im_h);
    
    % save the result
    strw = strcat('./results/SRLSP-',num2str(TestImgIndex),'_GR_regression_size-',num2str(patch_size),'_X',num2str(upscale),'.bmp');
    imwrite(im_SR,strw,'bmp');
    
    % display the objective results (PSNR and SSIM)
%     fprintf('\nPSNR for Nearest interpolation:   %f dB\n', nn_psnr(TestImgIndex));
%     fprintf('PSNR for Bicubic interpolation:   %f dB\n', bb_psnr(TestImgIndex));
%     fprintf('PSNR for the proposed method:     %f dB\n', sr_psnr(TestImgIndex));
%     fprintf('SSIM for Nearest interpolation:   %f \n', nn_ssim(TestImgIndex));
%     fprintf('SSIM for Bicubic interpolation:   %f \n', bb_ssim(TestImgIndex));
%     fprintf('SSIM for the proposed method:     %f \n', sr_ssim(TestImgIndex));

    % show the result
%     figure, subplot(1,4,1);imshow(im_n);
%     title('Nearest Interpolation');
%     xlabel({['PSNR = ',num2str(nn_psnr(TestImgIndex))]; ['SSIM = ',num2str(nn_ssim(TestImgIndex))]});
% 
%     subplot(1,4,2);imshow(im_b);
%     title('Bicubic Interpolation');
%     xlabel({['PSNR = ',num2str(bb_psnr(TestImgIndex))]; ['SSIM = ',num2str(bb_ssim(TestImgIndex))]});
% 
%     subplot(1,4,3);imshow(im_SR);
%     title('SRLSP Recovery');
%     xlabel({['PSNR = ',num2str(sr_psnr(TestImgIndex))]; ['SSIM = ',num2str(sr_ssim(TestImgIndex))]});
% 
%     subplot(1,4,4);imshow(im_h);
%     title('Original HR face');
    

end

fprintf('\n done!\n');
fprintf('===============================================\n');
fprintf('Average PSNR of Bicubic Interpolation: %f\n', sum(bb_psnr)/nTesting);
fprintf('Average PSNR of NN method: %f\n', sum(nn_psnr)/nTesting);
fprintf('Average PSNR of SRLSP method: %f\n', sum(sr_psnr)/nTesting);
fprintf('Average SSIM of Bicubic Interpolation: %f\n', sum(bb_ssim)/nTesting);
fprintf('Average SSIM of nn method: %f\n', sum(nn_ssim)/nTesting);
fprintf('Average SSIM of SRLSP method: %f\n', sum(sr_ssim)/nTesting);
fprintf('===============================================\n');

fprintf('sr_psnr =\n'); disp(sr_psnr)
fprintf('sr_ssim =\n'); disp(sr_ssim)

