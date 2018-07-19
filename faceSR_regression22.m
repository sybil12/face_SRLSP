
% =========================================================================
% local patch based global regression face super resolution
% version1.1 �ع���ѧϰ��SR֮ǰ��ʹ��row patch���м���
% version2.0 ѧϰ�ֲ��ֵ䣬����̫�� ��Ӧ��faceSR2.m
%=========================================================================


% clc; close all;
clear  % clear all;
addpath('./utilities');

% set parameters
% nrow       = 120;             % rows of HR face image
% ncol        = 100;            % cols of HR face image
% nTraining   = 321;              % number of training sample
% nTesting    = 35;                   % number of test sample
upscale     = 2;                    % upscaling factor
patch_size  = 8;                    % image patch (window) size
overlap     = patch_size-2;         % the overlap between neighborhood patches
center_flag = 1;             % '1': the missing HR pixels will not include the bordering
                             % '0': the missing HR pixels will include the bordering
impat_flag  = 1;             % to avoid changing the size of resulted HR image,
                             % we firstly expanded the input as well as the training set
impat_pixel = 2;             %% expandation level
% dict_size = 6;                  % num of atoms in a dict
% clusterszA = 240;               % num of patches used for counting projection matrix

% construct the HR and LR training pairs from the FEI face database
str = ['./data/YH_YL_FEI' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];
%str = ['./data/YH_YL_CMDP' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];

if exist(str,'file')
    load(str,'YH','YL');
else
%    trainpath = './CMDP_train/';
    trainpath = './FEI_train/';
    [YH, YL] = Training_myLH(upscale,trainpath,impat_flag,impat_pixel);
%     Training_myLH();
%     [YH, YL] = Training_LH(upscale,nTraining,impat_flag,impat_pixel); %double or im2double ����
    save(str,'YH','YL');
end
[imrow, imcol, ~] = size(YH);


lambda = 0.01;
%%  load or compute GR regression
str = ['./data/YH_YL_FEI_dict-PP' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];
% str = ['./data/YH_YL_CMDP_dict-PP' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];

if exist(str,'file')
    disp('Load trained GR-PPs...');
    load(str);
else
    disp('Training GR-PPs...');
    tic
    [PPs,dicts] = trainP_faceSR_for_dict(YH, YL, imrow, imcol, upscale,patch_size,overlap,impat_flag, lambda);
    toc
    save(str,'PPs','dicts')
    disp('Done.');
end
% dict_str = ['./data/YH_YL_FEI_' num2str(U) 'x' num2str(V) '_s' num2str(upscale) '_p' num2str(impat_flag) '_on_' num2str(dict_size) 'atoms_dicts.mat' ];
% load(str,'dicts');

%% testing
% testpath = './CMDP_test/';
testpath = './FEI_test/';
imgDir  = dir([testpath '*.bmp']); % ��������tif��ʽ�ļ�
nTesting = length(imgDir);
bb_psnr = zeros(1,nTesting);
bb_ssim = zeros(1,nTesting);
nn_psnr = zeros(1,nTesting);
nn_ssim = zeros(1,nTesting);
sr_psnr = zeros(1,nTesting);
sr_ssim = zeros(1,nTesting);
fprintf('\nface hallucinating _%dX for %d input test images using %d atoms_dicts\n', upscale, nTesting, size(dicts{1,1}.dl,2));
result_dir = ['./results/Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')];
mkdir(result_dir)

for TestImgIndex = 1:nTesting
    fprintf('\nProcessing  %d/%d LR image\n', TestImgIndex,nTesting);

    % read ground truth of one test face image
    imgpath = [testpath imgDir(TestImgIndex).name];
    im_h = imread(imgpath); %��ȡÿ��ͼƬ
    im_h = im2double(im_h);

    % use nearest method to get  lr img
    im_n = imresize(im_h,1/upscale,'nearest');
    im_l = im_n;
    if impat_flag == 1
        [im_l] = imrepat(im_l,impat_pixel/2);
    end
    im_l = im2double(im_l);

    % face SR
    tic
%     [im_SR] = faceSR(im_l,YH,YL,upscale,patch_size,overlap);
%     [im_SR] = faceSR2(im_l,PPs,dicts,imrow, imcol, upscale,patch_size,overlap);
    [im_SR] = faceSR2(im_l,PPs,dicts, imrow, imcol, upscale,patch_size,overlap,center_flag,impat_pixel);
    toc

    % bicubic interpolation for reference
%     im_b = imresize(im_l, [imrow,imcol], 'bicubic');
    im_b = imresize(im_l, upscale, 'bicubic');
    if impat_flag == 1
        im_b = imrepat(im_b,-impat_pixel);
        im_SR = imrepat(im_SR,-impat_pixel);
    end

    % nearest interpolation for reference
    im_n = imresize(im_n,upscale,'nearest');

%     im_h = uint8(im_h);im_n = uint8(im_n);im_b = uint8(im_b);im_SR = uint8(im_SR);
    im_h = im2uint8(im_h);im_n = im2uint8(im_n);im_b = im2uint8(im_b);im_SR = im2uint8(im_SR);
    disp('size of im_h im_l im_b im_n im_SR:')
    disp([size(im_h) size(im_l) size(im_b) size(im_n) size(im_SR)])

    % compute PSNR and SSIM for Bicubic and our method
    nn_psnr(TestImgIndex) = psnr(im_n,im_h);
    nn_ssim(TestImgIndex) = ssim(im_n,im_h);
    bb_psnr(TestImgIndex) = psnr(im_b,im_h);
    bb_ssim(TestImgIndex) = ssim(im_b,im_h);
    sr_psnr(TestImgIndex) = psnr(im_SR,im_h);
    sr_ssim(TestImgIndex) = ssim(im_SR,im_h);

    % save the result
    strw = strcat(result_dir,num2str(TestImgIndex),'_dictbased_patchsz',num2str(patch_size),'_X',num2str(upscale),'.bmp');
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

