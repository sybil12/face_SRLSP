function Y = Reshape3D(X,BlockSize)
tX = X(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4),:);

patch_size = BlockSize(2)+1-BlockSize(1);
patch_size = round(patch_size);  % patch_size should be int
Y = reshape(tX,patch_size*patch_size,size(X,3));