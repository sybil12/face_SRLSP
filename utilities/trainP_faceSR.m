function PPs = trainP_faceSR(YH, YL, imrow, imcol, upscale,patch_size,overlap, lambda)
%trainP_faceSR 此处显示有关此函数的摘要
%   此处显示详细说明 目标position的全部patch参与计算

%     % get high frequency of HR patches by interpolated reducing
%     for s = 1:size(YH,3)
%         img_l = YL(:,:,s);
%         interpolated = imresize(img_l, upscale, 'bicubic');
%         YH(:,:,s) = YH(:,:,s) - interpolated;
% %                     YH(:,:,s) = YH(:,:,s) - sum(sum(YH(:,:,s)))/(patch_size*patch_size);
%     end
            
    U = ceil((imrow-overlap)/(patch_size-overlap));
    V = ceil((imcol-overlap)/(patch_size-overlap));
    PPs = cell(U,V);
    for i = 1:U
        for j = 1:V
            BlockSize  =  GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);    
            BlockSizeS =  GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,i,j);
            BlockSize = floor(BlockSize);
            BlockSizeS = floor(BlockSizeS);
            

            Xh_patches   =  Reshape3D(YH,BlockSize);    % reshape each patch of HR face image to one column
            Xl_patches    =  Reshape3D(YL,BlockSizeS);   % reshape each patch of LR face image to one column
%             % get high frequency of HR patches by mean reducing
%             Xh_patches = Xh_patches - sum(Xh_patches)/size(Xh_patches,2);


            Xh_patches = double(Xh_patches');
            Xl_patches = double(Xl_patches');

            PP = ( Xl_patches'*Xl_patches + lambda*eye(size(Xl_patches,2)) ) \ Xl_patches';
            w = PP*Xh_patches;
            PPs{i,j} = w;
        end
    end
end

