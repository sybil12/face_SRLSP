% Returns a sampling grid for an image (using overlapping window)
%   GRID = SAMPLING_GRID(IMG_SIZE, WINDOW, OVERLAP, BORDER, SCALE)
% if area_flag=1, just collect features range in
% [area(1),area(2)]~[area(1)+area(3),area(2)+area(4)].
% % % %　img_size ＝［area(3),area(4)］
% % % %  area_left_corner = [area(1),area(2)]
function grid = sampling_grid(img_size, window, overlap, border, scale, area_flag, area)

if nargin < 7
    area_flag = 0;
    area = [1,1,img_size]; %取全部区域，从1开始，取整个size
end

if nargin < 5
    scale = 1;
end

if nargin < 4
    border = [0 0];   
end

if nargin < 3
    overlap = [0 0];    
end

% Scale all grid parameters
window = window * scale;
overlap = overlap * scale;
border = border * scale;

% Create sampling grid for overlapping window
index = reshape(1:prod(img_size), img_size);
grid = index(1:window(1), 1:window(2)) - 1;

% Compute offsets for grid's displacement.
skip = window - overlap; % for small overlaps
if area_flag==0
    offset = index(1+border(1) :skip(1): img_size(1)-window(1)+1-border(1), ...
               1+border(2) :skip(2): img_size(2)-window(2)+1-border(2));
elseif area_flag==1
     offset = index(area(1)+border(1) :skip(1): area(1)+area(3)-window(1)+1-border(1), ...
               area(2)+border(2) :skip(2): area(2)+area(4)-window(2)+1-border(2));
end
offset = reshape(offset, [1 1 numel(offset)]);

% Prepare 3D grid - should be used as: sampled_img = img(grid);
grid = repmat(grid, [1 1 numel(offset)]) + repmat(offset, [window 1]);

