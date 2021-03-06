
% =========================================================================
% Simple demo codes for face SR via SRLSP.
% For more details, pls refer to the following paper:
% J. Jiang, C. Chen, J. Ma, Z. Wang, Z. Wang, and R. Hu, ��SRLSP: A Face ImageSuper-Resolution Algorithm Using Smooth Regression with Local Structure Prior,��IEEETransactions on Multimedia, vol. 19, no. 1, pp. 27-40, 2017.
%=========================================================================

% clc; close all;
clear  % clear all;
addpath('./utilities');

% set parameters
nrow        = 120;           % rows of HR face image
ncol        = 100;           % cols of LR face image
nTraining   = 360;           % number of training sample
nTesting    = 40;             % number of test sample
upscale     = 2;             % upscaling factor
% BlurWindow  = 4;          % there is no blurring stage for SRLSP trainning img

center_flag = 1;             % '1': the missing HR pixels will not include the bordering
                             % '0': the missing HR pixels will include the bordering
impat_flag  = 1;             % to avoid changing the size of resulted HR image,
                                     % we firstly expanded the input as well as the training set
impat_pixel = 2;             %% expandation level???
alpha       = 1.2;         % the smooth regularization parameter in Equation (6)
patch_size  = 8;           % image patch (window) size
overlap     = patch_size-2;  % the overlap between neighborhood patches

% construct the HR and LR training pairs from the FEI face database
str = ['./data/YH_YL_FEI' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];
if exist(str,'file')
    load(str,'YH','YL');
else
    [YH, YL] = Training_LH(upscale,nTraining,impat_flag,impat_pixel); %double or im2double ����
    save(str,'YH','YL');
end

bb_psnr = zeros(1,nTesting);
bb_ssim = zeros(1,nTesting);
nn_psnr = zeros(1,nTesting);
nn_ssim = zeros(1,nTesting);
sr_psnr = zeros(1,nTesting);
sr_ssim = zeros(1,nTesting);
fprintf('\nface hallucinating _%dX for %d input test images\n', upscale, nTesting);

% super-resolve all the 40 test image, here you choose the specifical test image you want to process.
% for TestImgIndex = 1:1
for TestImgIndex = 1:nTesting
    fprintf('\nProcessing  %d/%d LR image\n', TestImgIndex,nTesting);

    % read ground truth of one test face image
    strh = strcat('./testFaces/',num2str(TestImgIndex),'_test.jpg');
%     strh    = strcat('.\testFaces\face',num2str(TestImgIndex),'.jpg');
    im_h = imread(strh);
    im_h = double(im_h);
%     im_h = im2double(im_h);

    % generate the input LR face image by smooth and down-sampleing
    im_n = imresize(im_h,1/upscale,'nearest');
    im_l = im_n;
    if impat_flag == 1
        [im_l] = imrepat(im_l,impat_pixel/2);
    end



    % face SR via SRLSP
    tic
    [im_SR] = SRLSP(im_l,YH,YL,upscale,patch_size,overlap,alpha,center_flag,impat_pixel);
    toc

    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow,ncol], 'bicubic');
    if impat_flag == 1
%         im_b = imrepat(im_b,-impat_pixel);
        im_SR = imrepat(im_SR,-impat_pixel);
    end


    % nearest interpolation for reference
    im_n = imresize(im_n,[nrow,ncol],'nearest');

%     im_h = im2uint8(im_h);
%     im_n = im2uint8(im_n);
%     im_b = im2uint8(im_b);
%     im_SR = im2uint8(im_SR);


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

    % save the result
    strw = strcat('./results/SRLSP-',num2str(TestImgIndex),'_alpha-',num2str(alpha),'_size-',num2str(patch_size),'_X',num2str(upscale),'.bmp');
%     im_SR = uint8(im_SR);
%     im_SR = im2uint8(im_SR);
    imwrite(im_SR,strw,'bmp');

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