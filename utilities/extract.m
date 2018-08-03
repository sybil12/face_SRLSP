function [features] = extract(conf, X, scale, filters, area_flag, area)
% if area_flag=1, just collect features range in
% [area(1),area(2)]~[area(1)+area(3),area(2)+area(4)].
% area work in extract->sampling_grid

if nargin < 6
    area_flag = 0;
    area = [];
end

% Compute one grid for all filters
grid = sampling_grid(size(X), ...
    conf.window, conf.overlap, conf.border, scale, area_flag, area);
feature_size = prod(conf.window) * numel(conf.filters);

% Current image features extraction [feature x index]
if isempty(filters)
    f = X(grid);
    features = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
else
    features = zeros([feature_size size(grid, 3)], 'single');
    for i = 1:numel(filters)
        f = conv2(X, filters{i}, 'same');
        f = f(grid);
        f = reshape(f, [size(f, 1) * size(f, 2) size(f, 3)]);
        features((1:size(f, 1)) + (i-1)*size(f, 1), :) = f;
    end
end
