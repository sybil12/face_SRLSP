function res = scaleup_area_ANR(conf, imgs)
% scaleup_ANR.m use "lr imgs"
% scaleup_area_ANR.m use "cropped interpolated imgs"

midres = imgs;
res = cell(1,numel(midres));
for i = 1:numel(midres)
    features = collect(conf, {midres{i}}, conf.upsample_factor, conf.filters);
    features = double(features);
    
    % Reconstruct using patches' dictionary and their anchored projections
    
    features = conf.V_pca'*features;
    
    patches = zeros(size(conf.PP,1),size(features,2));
    blocksize = 50000; %if not sufficient memory then you can reduce the blocksize
    if size(conf.dict_lores,2) > 10000
        blocksize = 500;
    end
    if size(features,2) < blocksize
        D = abs(conf.dict_lores'*features);
        [~, idx] = max(D);
        
        %if number of patches >> number of atoms in dictionary then you
        %can use the commented code for speed
        
        %             uidx = unique(idx);
        %             for u = 1: numel(uidx)
        %                 fidx = find(idx==uidx(u));
        %                 patches(:,fidx) = conf.PPs{uidx(u)}*features(:,fidx);
        %             end
        for l = 1:size(features,2)
            patches(:,l) = conf.PPs{idx(l)} * features(:,l);
        end
    else
        %% �ֿ���м���
        for b = 1:blocksize:size(features,2)
            if b+blocksize-1 > size(features,2)
                D = abs(conf.dict_lores'*features(:,b:end));
            else
                D = abs(conf.dict_lores'*features(:,b:b+blocksize-1));
            end
            [~, idx] = max(D);
            
            %                 uidx = unique(idx);
            %                 for u = 1: numel(uidx)
            %                     %fidx = find(idx==u);
            %                     fidx = find(idx==uidx(u));
            %                     patches(:,b-1+fidx) = conf.PPs{uidx(u)}*features(:,b-1+fidx);
            %                 end
            for l = 1:size(idx,2)
                patches(:,b-1+l) = conf.PPs{idx(l)} * features(:,b-1+l);
            end
            
        end
    end
    
    % Add low frequencies to each reconstructed patch
    patches = patches + collect(conf, {midres{i}}, conf.scale, {});
    
    % here img size is different from scaleup_ANR
    img_size = size(imgs{i});
%     grid = sampling_grid(img_size, ...
%         conf.window, conf.overlap, conf.border, conf.scale);
    grid = sampling_grid(img_size, ...
        conf.window, conf.overlap, conf.border, conf.scale);
    result = overlap_add(patches, img_size, grid);
    res{i} = result; % for the next iteration
    %         fprintf('.');
end
% fprintf('\n');
