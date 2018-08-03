function imgs = get_area(imgs,area,border)
if nargin<3
    border=0;
end

for i = 1:numel(imgs)
    imgs{i} = imgs{i}(area(1)-border+1:area(1)+area(3)+border,area(2)-border+1:area(2)+area(4)+border);
end

end

