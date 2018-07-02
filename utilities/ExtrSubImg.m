function subImg = ExtrSubImg(img,center_flag,impat_pixel,upscale)

% 保留LR像素点，并且边界位置也不做SR
% img(1:2:end,1:2:end) = -1;
img(1:upscale:end,1:upscale:end) = -1;

if center_flag == 1
    img([1:impat_pixel/2 end-(impat_pixel/2-1):end],:) = -1;
    img(:,[1:impat_pixel/2 end-(impat_pixel/2-1):end]) = -1;
end

% index = find(img~=-1);
% subImg = img(index);
subImg = img(img~=-1); %get HR pixels as a column, where using SR methods 