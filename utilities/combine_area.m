function imgs = combine_area(imgs,img_area,area,border)
if nargin<4
    border=0;
end

for i = 1:numel(imgs)
    imgs{i}(area(1)-border+1:area(1)+area(3)+border,area(2)-border+1:area(2)+area(4)+border) = img_area{i};
end

end

