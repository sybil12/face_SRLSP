function [YH, YL] = Training_myLH(upscale,trainpath,impat_flag,impat_pixel)
if nargin < 4
    impat_flag = 0;
    impat_pixel = 0;
end

% upscale     = 2;                    % upscaling factor
% trainpath = './CMDP_train/';
% str = ['./data/YH_YL_CMDP' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];
trainpath = './FEI_train/';
str = ['./data/YH_YL_EFI' '_s' num2str(upscale) '_p' num2str(impat_flag) '.mat'];


imgDir  = dir([trainpath '*.bmp']); % 遍历所有tif格式文件
% disp(length(imgDir))
img_size = size(imread([trainpath imgDir(1).name]));
YH = zeros(img_size(1)+impat_pixel*2, img_size(2)+impat_pixel*2, length(imgDir));
YL = zeros(img_size(1)/upscale+impat_pixel, img_size(2)/upscale+impat_pixel, length(imgDir));

if ~exist(str,'file')
    disp('Constructing the HR-LR training set...');
    for i = 1:length(imgDir)          % 遍历结构体就可以一一处理图片了
        HI = imread([trainpath imgDir(i).name]); %读取每张图片
        HI = im2double(HI);
        if impat_flag == 1
            [HIP] = imrepat(HI,impat_pixel);
            YH(:,:,i) = HIP;
        else
            YH(:,:,i) = HI;
        end

        %%% generate the LR face image by smooth and down-sampling
        LI = imresize(HI,1/upscale,'nearest');
        if impat_flag == 1
            [LI] = imrepat(LI,impat_pixel/2);
        end
        YL(:,:,i) = im2double(LI);
%         imshowpair(HI, LI, 'montage')
    end
%     save(str,'YH','YL');
    disp('done.');
else
    disp(['training file ' str  ' exist!']);
end