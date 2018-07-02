function [PPs,dicts] = trainP_faceSR_for_dict(YH, YL, imrow, imcol, upscale,patch_size,overlap, lambda)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明

%     % get high frequency of HR patches by interpolated reducing
%     for s = 1:size(YH,3)
%         img_l = YL(:,:,s);
%         interpolated = imresize(img_l, upscale, 'bicubic');
%         YH(:,:,s) = YH(:,:,s) - interpolated;
% %                     YH(:,:,s) = YH(:,:,s) - sum(sum(YH(:,:,s)))/(patch_size*patch_size);
%     end

    U = ceil((imrow-overlap)/(patch_size-overlap));
    V = ceil((imcol-overlap)/(patch_size-overlap));

    dict_size = 36;
    clusterszA = 100;
    alpha       = 1.2;
    dict_str = ['./data/YH_YL_FEI_' num2str(U) 'x' num2str(V) '_on_' num2str(dict_size) 'atoms_dicts.mat' ];
    dict_flag = 0;
    if exist(dict_str,'file')
        dict_flag = 1;
        load(dict_str,'dicts');
    else
        dicts = cell(U,V);
    end

    PPs = cell(U,V);
    for i = 1:U
        fprintf('.')
        for j = 1:V
            BlockSize  =  GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);
            BlockSizeS =  GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,i,j);
            BlockSize = floor(BlockSize);
            BlockSizeS = floor(BlockSizeS);


            Xh_patches   =  Reshape3D(YH,BlockSize);    % reshape each patch of HR face image to one column
            Xl_patches    =  Reshape3D(YL,BlockSizeS);   % reshape each patch of LR face image to one column

            %  l2 normalize LR patches to compute distance
            l2 = sum(Xl_patches.^2).^0.5 + eps;
            l2 = repmat(l2,size(Xl_patches,1),1);
            lrps_l2 = Xl_patches./l2;
            clear l2;

            if dict_flag
                dl = dicts{i,j}.dl;
            else
                fprintf('computing (%d,%d) position dict....\n',i,j)
                dict = train_local_based_dict(Xh_patches, Xl_patches,dict_size);
                dl = dict.dl; %dh = dicts.dh;
                dicts{i,j} = dict;
            end

            %% use all patches at current position,but add locality constain, to compute projection matrix for each atom
            PPs{i,j} = cell(1,dict_size);
            for  d= 1:size(dl,2)
                dl_atom = dl(d);

                D = pdist2(single(lrps_l2'),single(dl(:,d)'));  %Distance matrix, use Euclidean  /return a cloumn vector
                [~, idx] = sort(D);
                Lo = Xl_patches(:, idx(1:clusterszA));
                Hi = Xh_patches(:, idx(1:clusterszA));
                Dis = sum((repmat(dl_atom,1,clusterszA) - lrps_l2(:, idx(1:clusterszA))).^2);

%                 Lo = Xl_patches;
%                 Hi = Xh_patches;
%                 Dis = sum((repmat(dl_atom,1,size(lrps_l2,2)) - lrps_l2).^2);
                Dis = 1./(Dis+1e-6).^alpha;
                PPs{i, j}{d} = Hi*diag(Dis)*Lo'*pinv(Lo*diag(Dis)*Lo'+1e-9*eye(size(Lo,1)));       %pinv
            end

            %% use clusterszA most close patches at current position to compute projection matrix for each atom
%             PPs{i,j} = cell(1,dict_size);
%             for  d= 1:size(dl,2)
%                 D = pdist2(single(lrps_l2'),single(dl(:,d)'));  %Distance matrix, use Euclidean  /return a cloumn vector
%                 [~, idx] = sort(D);
%                 Lo = Xl_patches(:, idx(1:clusterszA));
%                 Hi = Xh_patches(:, idx(1:clusterszA));
%                 PPs{i, j}{d} = Hi/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';
%             end


            %%  GR method: using all patches at current position to compute projection matrix
%             Lo = Xl_patches;
%             Hi = Xh_patches;
%             PPs{i, j} = Hi/(Lo'*Lo+lambda*eye(size(Lo,2)))*Lo';

               %和前三行等效
%             Xh_patches = double(Xh_patches');
%             Xl_patches = double(Xl_patches');
%
%             PP = ( Xl_patches'*Xl_patches + lambda*eye(size(Xl_patches,2)) ) \ Xl_patches';
%             w = PP*Xh_patches;
%             PPs{i,j} = w;
        end
    end
    if dict_flag==0
        save(dict_str,'dicts')
    end
end

