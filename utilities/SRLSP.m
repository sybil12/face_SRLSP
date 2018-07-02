
function im_SR = SRLSP(im_l,YH,YL,upscale,patch_size,overlap,alpha,center_flag,impat_pixel)

[imrow, imcol , ~] = size(YH);

Img_SUM      = zeros(imrow,imcol);
overlap_FLAG = zeros(imrow,imcol);

U = ceil((imrow-overlap)/(patch_size-overlap));
V = ceil((imcol-overlap)/(patch_size-overlap));

sub_flag = zeros(patch_size-1);
% super-resolve the HR image patch by patch
for i = 1:U
   fprintf('.');
   for j = 1:V

        BlockSize = GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);
        BlockSize(1) = BlockSize(1)+1; BlockSize(3) = BlockSize(3)+1;
        BlockSizeS = GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,i,j);
        BlockSize = floor(BlockSize);
        BlockSizeS = floor(BlockSizeS);

        im_l_patch = im_l(BlockSizeS(1):BlockSizeS(2),BlockSizeS(3):BlockSizeS(4));           % extract the patch at position（i,j）of the input LR face
%         imshow(im_l_patch)
        im_l_patch = im_l_patch(:);   % Reshape 2D image patch into 1D column vectors

%         XF = ExtrImg3D(YH,BlockSize,center_flag,impat_pixel,upscale);    % reshape each patch of HR face image to one column
        %B2-不保留原像素点
        XF  = Reshape3D(YH,BlockSize);
        X  = Reshape3D(YL,BlockSizeS);   % reshape each patch of LR face image to one column

        % smooth regression
        Dis = sum((repmat(im_l_patch,1,size(X,2))-X).^2);
        Dis = 1./(Dis+1e-6).^alpha;
        P = XF*diag(Dis)*X'*pinv(X*diag(Dis)*X'+1e-9*eye(size(X,1)));       %pinv
        P_patch = P*im_l_patch;

        % obtain the HR patch
%         % 保留LR像素点
%         Img_flag = zeros(patch_size-1,patch_size-1);
%         Img_flag(1:upscale:end,1:upscale:end) = -1;
% 
%         Img = zeros(patch_size-1,patch_size-1);
%         Img(Img_flag==-1) = im_l_patch;  %if Img doesn't init,then it has the same size as Img_flag
%         sub_flag(Img_flag==-1) = 1; %原LR像素点也参与求平均
% 
%         if center_flag == 1
%             Img_flag([1:impat_pixel/2 end-(impat_pixel/2-1):end],:) = -1;
%             Img_flag(:,[1:impat_pixel/2 end-(impat_pixel/2-1):end]) = -1;
%         end
%         % 计算得到的SR像素点
%         Img(Img_flag~=-1) = P_patch;
%         sub_flag(Img_flag~=-1) = 1;

        %B2-不保留LR像素点 所有像素都计算
        Img = P_patch;

        % integrate all the LR patch
        Img = reshape(Img,patch_size-1,patch_size-1);
        Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))      = Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+Img;
%         overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+sub_flag;
        %B2
        overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+1;
    end
end
%  averaging pixel values in the overlapping regions
im_SR = Img_SUM./overlap_FLAG;
% im_SR = uint8(im_SR);
