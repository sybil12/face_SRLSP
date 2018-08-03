function [features] = collect(conf, imgs, scale, filters, verbose, area_flag, area) 
% if area_flag=1, just collect features range in
% [area(1),area(2)]~[area(1)+area(3),area(2)+area(4)].
% area work in collet->extract->sampling_grid
% used in learn_area_dict.m(deleted), replaced by get_area before collect.m

if nargin < 5
    verbose = 0;
end
if nargin < 7
    area_flag = 0;
    area = [];
end

num_of_imgs = numel(imgs);
feature_cell = cell(num_of_imgs, 1); % contains images' features
num_of_features = 0;

if verbose
    fprintf('Collecting features from %d image(s) ', num_of_imgs)
end
feature_size = [];

% h = [];
for i = 1:num_of_imgs
%     h = progress(h, i / num_of_imgs, verbose);
    sz = size(imgs{i});
    if verbose
        fprintf(' [%d x %d]', sz(1), sz(2));
    end
    
    F = extract(conf, imgs{i}, scale, filters, area_flag, area);
    num_of_features = num_of_features + size(F, 2);
    feature_cell{i} = F;

    assert(isempty(feature_size) || feature_size == size(F, 1), ...
        'Inconsistent feature size!')
    feature_size = size(F, 1);
end
if verbose
    fprintf('\nExtracted %d features (size: %d)\n', num_of_features, feature_size);
end
clear imgs % to save memory
features = zeros([feature_size num_of_features], 'single');
offset = 0;
for i = 1:num_of_imgs
    F = feature_cell{i};
    N = size(F, 2); % number of features in current cell
    features(:, (1:N) + offset) = F;
    offset = offset + N;
end
