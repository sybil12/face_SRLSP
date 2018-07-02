function Y = ExtrImg3D(X,BlockSize,center_flag,impat_pixel,upscale)

tX = X(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4),:);

temp = ExtrSubImg(tX(:,:,1),center_flag,impat_pixel,upscale);
len = length(temp);
% X1 = [];
X1 = zeros(len,size(tX,3));
for i = 1:size(tX,3)
    X1(:,i) = ExtrSubImg(tX(:,:,i),center_flag,impat_pixel,upscale);
%     X1 = [X1 ExtrSubImg(tX(:,:,i),center_flag,impat_pixel,upscale)];
end
Y = double(X1);