function im_SR = faceSR(im_l,PPs,imrow, imcol, upscale,patch_size,overlap)

%faceSR 此处显示有关此函数的摘要
%   此处显示详细说明

% [imrow, imcol , ~] = size(YH);
U = ceil((imrow-overlap)/(patch_size-overlap));
V = ceil((imcol-overlap)/(patch_size-overlap));


Img_SUM      = zeros(imrow,imcol);
overlap_FLAG = zeros(imrow,imcol);



% hallucinate the HR image patch by patch
for i = 1:U
   fprintf('.');
   for j = 1:V
       
        BlockSize  =  GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);    
        BlockSizeS =  GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,i,j);
        BlockSize = floor(BlockSize);
        BlockSizeS = floor(BlockSizeS);
       
        im_l_patch = im_l(BlockSizeS(1):BlockSizeS(2),BlockSizeS(3):BlockSizeS(4));           % extract the patch at position（i,j）of the input LR face
        im_l_patch = im_l_patch(:);
        
%         Xh_patches   =  Reshape3D(YH,BlockSize);    % reshape each patch of HR face image to one column
%         Xl_patches    =  Reshape3D(YL,BlockSizeS);   % reshape each patch of LR face image to one column
%         Xh_patches = double(Xh_patches');
%         Xl_patches = double(Xl_patches');
%                 
%         PP = ( Xl_patches'*Xl_patches + lambda*eye(size(Xl_patches,2)) ) \ Xl_patches';
%         w = PP*Xh_patches;

        w = PPs{i,j};
        Img  =  im_l_patch'*w;
              
        Img  =  reshape(Img,patch_size,patch_size);
%         imshow(Img)
%         % only use SR to get high frequency details
%         Img = Img + imresize(im_l_patch,[patch_size,patch_size],'nearest');
%         Img = Img + imresize(im_l_patch,[patch_size,patch_size],'bicubic');

        % integrate all the LR patch 
        Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))      = Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+Img;
        overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+1;
        
   end
end
%  averaging pixel values in the overlapping regions
im_SR = Img_SUM./overlap_FLAG;

end

