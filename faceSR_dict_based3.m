% Anchored Neighborhood Regression for Face Super-Resolution
% using Bicubic, GR, ANR, A+(0.5 mil), A+.
% version2.0 add locality constrain
% version3.0 add area dictionary [using learn_area_dict.m]

%clc
%clear;
addpath('./utilities');
verbose = 0;

%       % make sure you have make ompbox and ksvdbox already,
%       % if not, make it and before you should have a gcc compiler
%       % then use "addpath" add toolbox path

imgscale = 1; % the scale reference we work with
flag = 0;       % flag = 0 - only GR, ANR, A+, and bicubic methods, the other get the bicubic result by default
% flag = 1 - all the methods are applied
area_flag = 1;  % area_flag = 1 - use area dict

upscaling = 2; % the magnification factor x2, x3, x4...
train_dir = 'train_faces';
% train_dir = 'train_faces';
train_pattern = '*.bmp';
input_dir = 'FEI_test';
input_pattern = '*.bmp';

dict_sizes = [2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];

clusterszA = 2048; % neighborhood size for A+

if flag==1
    disp('All methods are employed : Bicubic, Yang et al., Zeyde et al., GR, ANR, A+(0.5 mil), A+.');
else
    disp('We run only for Bicubic, GR, ANR and A+ methods, the other get the Bicubic result by default.');
end

%%
for d=10    %1024 for dict size
    %% load or learn area dict, and compute projections
    area_dict_size = 512;
    ares_train_dir = 'train_faces';
    area_dict = ['conf_Zeyde_' ares_train_dir '_' num2str(area_dict_size) '_finalx' num2str(upscaling) '_area'];
    if exist([area_dict '.mat'],'file')
        disp(['Load trained dictionary...' area_dict]);
        load([area_dict '.mat'], 'confeye','confmouth');
    else
        disp(['Training area_dictionary of size ' num2str(area_dict_size) ' for ' ares_train_dir ' dataset using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [4 4]; % low-res. window size
        conf.border = [1 1]; % border of the lr image (to ignore) %inner area has no border?
        
        % High-pass filters for feature extraction (defined for upsampled low-res.)
        % here is going to use LBP CODE
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';
        
        conf.overlap = [1 1]; % partial overlap (for faster training)
        eye_area=[24,10,30,84];
        mouth_area=[54,24,50,54];
        
        startt = tic;
        confeye = learn_dict(conf, ...
            get_area( load_images(glob( ares_train_dir, train_pattern) ), eye_area),...
            area_dict_size);
        confmouth = learn_dict(conf, ...
            get_area( load_images(glob(ares_train_dir, train_pattern) ), mouth_area),...
            area_dict_size);
        toc(startt)
        
        confeye.area=eye_area;
        confmouth.area=mouth_area;
        confeye.overlap = confeye.window - [1 1]; % full overlap scheme (for better reconstruction)
        confmouth.overlap = confmouth.window - [1 1]; % full overlap scheme (for better reconstruction)
        save([area_dict '.mat'], 'confeye', 'confmouth');
        
        % train call
    end
    
    
    % GR  PP  and  ANR_PPs
    fname = [area_dict '_ANR_projections_imgscale_' num2str(imgscale) '.mat'];
    if exist(fname,'file')  %load A+ regressors
        disp(['Load area ANR_projections...' fname]);
        load(fname);
    else  % count ProjM and PP
        %%
        disp('compute area ANR_projections...')
        lambda = 0.01;
        confeye.ProjM = (confeye.dict_lores'*confeye.dict_lores+lambda*eye(size(confeye.dict_lores,2)))\confeye.dict_lores';
        confeye.PP = (1+lambda)*confeye.dict_hires*confeye.ProjM;
        confmouth.ProjM = (confmouth.dict_lores'*confmouth.dict_lores+lambda*eye(size(confmouth.dict_lores,2)))\confmouth.dict_lores';
        confmouth.PP = (1+lambda)*confmouth.dict_hires*confmouth.ProjM;
        
        % precompute for ANR the anchored neighborhoods and the projection matrices for the dictionary
        % count PPs , project matrix when uses K neighbours for each atom
        
        confeye.PPs = [];
        % clustersz , the K in paper , num of neighbours
        if  size(confeye.dict_lores,2) < 30
            clustersz = size(confeye.dict_lores,2);
        else
            clustersz = 30;
        end
        D = abs(confeye.dict_lores'*confeye.dict_lores);  %D-- Correlation matrix
        
        % each cloumn in PPs is correspond to a atom in D_l
        for i = 1:size(confeye.dict_lores,2)
            [vals, idx] = sort(D(i,:), 'descend');  % idx represent the origin index in i row of D
            if (clustersz >= size(confeye.dict_lores,2)/2)
                confeye.PPs{i} = confeye.PP;
            else
                Lo = confeye.dict_lores(:, idx(1:clustersz));
                confeye.PPs{i} = 1.01*confeye.dict_hires(:,idx(1:clustersz))/(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';
            end
        end
        
        confmouth.PPs = [];
        % clustersz , the K in paper , num of neighbours
        if  size(confmouth.dict_lores,2) < 30
            clustersz = size(confmouth.dict_lores,2);
        else
            clustersz = 30;
        end
        D = abs(confmouth.dict_lores'*confmouth.dict_lores);  %D-- Correlation matrix
        
        % each cloumn in PPs is correspond to a atom in D_l
        for i = 1:size(confmouth.dict_lores,2)
            [vals, idx] = sort(D(i,:), 'descend');  % idx represent the origin index in i row of D
            if (clustersz >= size(confmouth.dict_lores,2)/2)
                confmouth.PPs{i} = confmouth.PP;
            else
                Lo = confmouth.dict_lores(:, idx(1:clustersz));
                confmouth.PPs{i} = 1.01*confmouth.dict_hires(:,idx(1:clustersz))/(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';
            end
        end
        
        confeye.fname = fname;
        confmouth.fname = fname;
        save(fname,'confeye','confmouth');
        
    end
    eye_ANR_PPs = confeye.PPs;
    mouth_ANR_PPs = confmouth.PPs;
    
    %%　computing a+ regressors for area(aye for 120w )
    fname = [area_dict '_Aplus_x' num2str(upscaling) '_' num2str(area_dict_size) 'atoms' num2str(clusterszA) 'nn.mat'];
    if exist(fname,'file')  %load A+ regressors
        disp(['Load Aplus_projections...' fname]);
        load(fname);
    else  %Compute A+ regressors
        %%
        disp('Compute area A+ regressors');
        ttime = tic;
        [plores, phires] = collectSamplesScales(confeye, get_area(load_images(...
            glob(train_dir, train_pattern)),eye_area), 12, 0.98);
        
        if size(plores,2) > 5000000
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples_eye = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);
        l2(l2<0.1) = 1;
        disp(['num of l2<0.1: ' num2str(sum(l2<0.1))])
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n
        
        aalpha = 1.2;
        Aplus_PPs_eye = cell(size(confeye.dict_lores,2),1);
        % count projection matrix with K=clusterszA neighborhood patches
        for i = 1:size(confeye.dict_lores,2)
            D = pdist2(single(plores'),single(confeye.dict_lores(:,i)'));  %Distance matrix, use Euclidean
            [~, idx] = sort(D);
            Lo = plores(:, idx(1:clusterszA));
            Hi = phires(:, idx(1:clusterszA));
            %             Aplus_PPs{i} = Hi/(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo';
            Dis = D(idx(1:clusterszA));
            Dis = 1./(Dis+1e-6).^aalpha;
            Aplus_PPs_eye{i} = Hi*diag(Dis)*Lo'*pinv(Lo*diag(Dis)*Lo'+1e-9*eye(size(Lo,1)));       %pinv
        end
        clear plores
        clear phires
        
        [plores, phires] = collectSamplesScales(confmouth, get_area(load_images(...
            glob(train_dir, train_pattern)),mouth_area), 12, 0.98);
        
        if size(plores,2) > 5000000
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples_mouth = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);
        l2(l2<0.1) = 1;
        disp(['num of l2<0.1: ' num2str(sum(l2<0.1))])
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n
        
        llambda = 0.1;aalpha = 1.2;
        Aplus_PPs_mouth = cell(size(confmouth.dict_lores,2),1);
        % count projection matrix with K=clusterszA neighborhood patches
        for i = 1:size(confmouth.dict_lores,2)
            D = pdist2(single(plores'),single(confmouth.dict_lores(:,i)'));  %Distance matrix, use Euclidean
            [~, idx] = sort(D);
            Lo = plores(:, idx(1:clusterszA));
            Hi = phires(:, idx(1:clusterszA));
            %             Aplus_PPs{i} = Hi/(Lo'*Lo+llambda*mouth(size(Lo,2)))*Lo';
            Dis = D(idx(1:clusterszA));
            Dis = 1./(Dis+1e-6).^aalpha;
            Aplus_PPs_mouth{i} = Hi*diag(Dis)*Lo'*pinv(Lo*diag(Dis)*Lo'+1e-9*eye(size(Lo,1)));       %pinv
        end
        clear plores
        clear phires
        
        ttime = toc(ttime);
        disp(['time passing ' num2str(ttime) ' s'])
        save(fname,'Aplus_PPs_eye', 'Aplus_PPs_mouth','number_samples_eye', 'number_samples_mouth', 'ttime', 'fname');
        toc
    end
    
    
    %%  tag--result mat_file--train_dict
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    %     mat_file = ['conf_Zeyde_CMDP' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];
    mat_file = ['conf_Zeyde_' train_dir '_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];
    
    %% load or learn dict
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file '.mat']);
        load([mat_file '.mat'], 'conf');
    else
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' for ' train_dir ' dataset using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [4 4]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)
        
        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';
        
        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2
            conf.overlap = [1 1]; % partial overlap (for faster training)
        end
        
        startt = tic;
        conf = learn_dict(conf, ...
            load_images(glob(train_dir, train_pattern) ),...
            dict_sizes(d));
        
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)
        conf.trainingtime = toc(startt);
        toc(startt)
        conf.mat_file=mat_file;
        save(mat_file, 'conf');
        
        % train call
    end
    
    %% set lambda
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end
    
    %% conf
    %method name
    conf.desc = {'Original', 'Bicubic', 'Yang', 'Zeyde',...
        'GR', 'ANR', 'A+(0.5mil)','A+'};
    
    % index of atoms
    conf.points = 1:1:size(conf.dict_lores,2);
    conf.pointslo = conf.dict_lores(:,conf.points);
    %     conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    
    
    %% GR  PP  and  ANR_PPs
    fname = [mat_file '_ANR_projections_imgscale_' num2str(imgscale) '.mat'];
    if exist(fname,'file')  %load A+ regressors
        disp(['Load ANR_projections...' fname]);
        load(fname);
    else  % count ProjM and PP
        if dict_sizes(d) < 10000
            conf.ProjM = (conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))\conf.dict_lores';
            conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
        else
            % here should be an approximation
            conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
            conf.ProjM = [];
        end
        
        % precompute for ANR the anchored neighborhoods and the projection matrices for the dictionary
        
        % count PPs , project matrix when uses K neighbours for each atom
        conf.PPs = [];
        
        % clustersz , the K in paper , num of neighbours
        if  size(conf.dict_lores,2) < 40
            clustersz = size(conf.dict_lores,2);
        else
            clustersz = 40;
        end
        D = abs(conf.pointslo'*conf.dict_lores);  %D-- Correlation matrix
        
        % each cloumn in PPs is correspond to a atom in D_l
        for i = 1:length(conf.points)
            [vals, idx] = sort(D(i,:), 'descend');  % idx represent the origin index in i row of D
            if (clustersz >= size(conf.dict_lores,2)/2)
                conf.PPs{i} = conf.PP;
            else
                Lo = conf.dict_lores(:, idx(1:clustersz));
                conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))/(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';
            end
        end
        conf.fname = fname;
        save(fname,'conf');
    end
    ANR_PPs = conf.PPs;
    
    %% A+ (5 mil)computing the regressors numscales(12) and scalefactor(0.98)
    Aplus_PPs = [];
    fname = [mat_file '_Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_5mil.mat'];
    
    if exist(fname,'file')  %load A+ regressors
        disp(['Load Aplus_projections...' fname]);
        load(fname);
    else  %Compute A+ regressors
        %%
        disp('Compute A+ regressors');
        ttime = tic;
        tic
        [plores, phires] = collectSamplesScales(conf, load_images(...
            glob(train_dir, train_pattern)), 12, 0.98);
        
        if size(plores,2) > 5000000
            plores = plores(:,1:5000000);
            phires = phires(:,1:5000000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);
        l2(l2<0.1) = 1;
        disp(['num of l2<0.1: ' num2str(sum(l2<0.1))])
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n
        
        llambda = 0.1;aalpha = 1.2;
        Aplus_PPs = cell(size(conf.dict_lores,2),1);
        % count projection matrix with K=clusterszA neighborhood patches
        for i = 1:size(conf.dict_lores,2)
            D = pdist2(single(plores'),single(conf.dict_lores(:,i)'));  %Distance matrix, use Euclidean
            [~, idx] = sort(D);
            Lo = plores(:, idx(1:clusterszA));
            Hi = phires(:, idx(1:clusterszA));
            %             Aplus_PPs{i} = Hi/(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo';
            Dis = D(idx(1:clusterszA));
            Dis = 1./(Dis+1e-6).^aalpha;
            Aplus_PPs{i} = Hi*diag(Dis)*Lo'*pinv(Lo*diag(Dis)*Lo'+1e-9*eye(size(Lo,1)));       %pinv
        end
        clear plores
        clear phires
        
        ttime = toc(ttime);
        save(fname,'Aplus_PPs','ttime', 'number_samples','fname');
        toc
    end
    
    %% A+ (0.5 mil)computing the regressors with 0.5 milion training samples
    % numscales(1) and scalefactor(1)
    Aplus05_PPs = [];
    fname = [mat_file '_Aplus_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms' num2str(clusterszA) 'nn_05mil.mat'];
    
    if exist(fname,'file') %load A+(0.5 mil) regressors
        disp(['Load Aplus05_projections...' fname]);
        load(fname);
    else   %Compute A+(0.5 mil) regressors
        %%
        disp('Compute A+ (0.5 mil) regressors');
        ttime = tic;
        tic
        [plores, phires] = collectSamplesScales(conf, load_images(...
            glob(train_dir, train_pattern)), 1, 1);
        
        if size(plores,2) > 500000
            idx_random = randperm(size(plores,2));
            plores = plores(:,idx_random(1:500000));
            phires = phires(:,idx_random(1:500000));
            clear idx_random
            %             plores = plores(:,1:500000);
            %             phires = phires(:,1:500000);
        end
        number_samples = size(plores,2);
        
        % l2 normalize LR patches, and scale the corresponding HR patches
        l2 = sum(plores.^2).^0.5+eps;
        l2n = repmat(l2,size(plores,1),1);
        l2(l2<0.1) = 1;
        plores = plores./l2n;
        phires = phires./repmat(l2,size(phires,1),1);
        clear l2
        clear l2n
        
        llambda = 0.1;aalpha = 1.2;
        Aplus05_PPs = cell(size(conf.dict_lores,2),1);
        % count projection matrix with K=clusterszA neighborhood patches
        for i = 1:size(conf.dict_lores,2)
            D = pdist2(single(plores'),single(conf.dict_lores(:,i)'));  %Distance matrix, use Euclidean
            [~, idx] = sort(D);
            Lo = plores(:, idx(1:clusterszA));
            Hi = phires(:, idx(1:clusterszA));
            %             Aplus05_PPs{i} = Hi/(Lo'*Lo+llambda*eye(size(Lo,2)))*Lo';
            Dis = D(idx(1:clusterszA));
            Dis = 1./(Dis+1e-6).^aalpha;
            Aplus05_PPs{i} = Hi*diag(Dis)*Lo'*pinv(Lo*diag(Dis)*Lo'+1e-9*eye(size(Lo,1)));       %pinv
        end
        clear plores
        clear phires
        
        ttime = toc(ttime);
        save(fname,'Aplus05_PPs','ttime', 'number_samples','fname');
        toc
    end
    
    %% conf
    
    % get all img names from the dir
    conf.filenames = glob(input_dir, input_pattern); % Cell array
    
    conf.results = {};
    %     conf.result_dir = ['./results/Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')];
    %     mkdir(conf.result_dir)
    if numel(conf.filenames)>0
        conf.result_dir = qmkdir(['./results/Results-' input_dir '_' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
        conf.result_dirImages = qmkdir([conf.result_dir '/results_' tag]);
        conf.result_dirImagesRGB = qmkdir([conf.result_dir '/results_' tag 'RGB']);
    else
        disp( ['  pay attention:   ' input_dir ' has no ' input_pattern ' imgs !!!'] )
        break
    end
    
    conf.countedtime = zeros(numel(conf.filenames), numel(conf.desc));
    PSNRs = zeros(numel(conf.filenames), numel(conf.desc)-1);
    SSIMs = zeros(numel(conf.filenames), numel(conf.desc)-1);
    
    fprintf('\n')
    disp(['Upscaling ' num2str(numel(conf.filenames)) ' imgs in '  input_dir ' x' num2str(upscaling) ' with Zeyde dictionary of size = ' num2str(dict_sizes(d))]);
    
    %% upscaling i_th image
    res =[];
    % area_eye,res_eye,area_mouth,res_mouth,face
    psnr_anr = zeros(numel(conf.filenames,5));
    psnr_aplus = zeros(numel(conf.filenames,5));
    ssim_anr = zeros(numel(conf.filenames,5));
    ssim_aplus = zeros(numel(conf.filenames,5));
    fsim_anr = zeros(numel(conf.filenames,5));
    fsim_aplus = zeros(numel(conf.filenames,5));
    t = cputime;
    %%
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f});
        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
        
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);
        
        low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
            lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
        end
        
        %cheap upscaling
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);
        end
        
        res{1} = interpolated;
        
        %       if (flag == 1) && (dict_sizes(d) == 1024) && (upscaling==3)
        %           startt = tic;
        %           res{2} = {yima(low{1}, upscaling)};
        %           tt = toc(startt);
        %           conf.countedtime(i,2) = toc(startt);
        %           if verbose==1
        %               disp(['时间已过' num2str(conf.countedtime(i,2)) '秒'])
        %           end
        %       else
        res{2} = interpolated;
        %       end
        
        %       if (flag == 1)
        %           startt = tic;
        %           res{3} = scaleup_Zeyde(conf, low);
        %           conf.countedtime(i,3) = toc(startt);
        %           if verbose==1
        %               disp(['时间已过' num2str(conf.countedtime(i,3)) '秒'])
        %           end
        %       else
        res{3} = interpolated;
        %       end
        
        %if flag == 1
        startt = tic;
        res{4} = scaleup_GR(conf, low);
        conf.countedtime(i,4) = toc(startt);
        if verbose==1
            disp(['时间已过' num2str(conf.countedtime(i,4)) '秒'])
        end
        %else
        %res{4} = interpolated;
        %end
        
        startt = tic;
        conf.PPs = ANR_PPs;
        res{5} = scaleup_ANR(conf, low);
        if area_flag==1         % area scaleup
            eyes = get_area(interpolated, confeye.area, confeye.border*confeye.scale);
            eyes = scaleup_area_ANR(confeye, eyes);
            eyes = shave(eyes{1}, confeye.border*confeye.scale);
            res_eye = get_area(res{5}, confeye.area);res_eye=res_eye{1};
            row_eye = get_area(img, confeye.area);row_eye=row_eye{1};
            psnr_anr(i,1) = psnr(double(eyes),double(row_eye));
            psnr_anr(i,2) = psnr(double(res_eye),double(row_eye));
            ssim_anr(i,1) = ssim(double(eyes),double(row_eye));
            ssim_anr(i,2) = ssim(double(res_eye),double(row_eye));
            fsim_anr(i,1) = FeatureSIM(double(eyes),double(row_eye));
            fsim_anr(i,2) = FeatureSIM(double(res_eye),double(row_eye));
            %             disp(['psnr for area_eye dict: ' num2str(psnr(double(eyes),double(row_eye)))]);
            %             disp(['psnr for face dict in eye_area: ' num2str(psnr(double(res_eye),double(row_eye)))]);
            %             disp(['psnr for face dict: ' num2str(psnr( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) ))]);
            
            mouth = get_area(interpolated, confmouth.area, confeye.border*confeye.scale);
            mouth = scaleup_area_ANR(confeye, mouth);
            mouth = shave(mouth{1}, confeye.border*confeye.scale);
            res_mouth = get_area(res{5}, confmouth.area);res_mouth=res_mouth{1};
            row_mouth = get_area(img, confmouth.area);row_mouth=row_mouth{1};
            psnr_anr(i,3) = psnr(double(mouth),double(row_mouth));
            psnr_anr(i,4) = psnr(double(res_mouth),double(row_mouth));
            psnr_anr(i,5) = psnr( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) );
            ssim_anr(i,3) = ssim(double(mouth),double(row_mouth));
            ssim_anr(i,4) = ssim(double(res_mouth),double(row_mouth));
            ssim_anr(i,5) = ssim( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) );
            fsim_anr(i,3) = FeatureSIM(double(mouth),double(row_mouth));
            fsim_anr(i,4) = FeatureSIM(double(res_mouth),double(row_mouth));
            fsim_anr(i,5) = FeatureSIM( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) );
            
            %             disp(['psnr for area_mouth dict: ' num2str(psnr(double(mouth),double(row_mouth)))]);
            %             disp(['psnr for face dict in mouth_area: ' num2str(psnr(double(res_mouth),double(row_mouth)))]);
            %             disp(['psnr for face dict: ' num2str(psnr( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) ))]);
            
        end
        
        conf.countedtime(i,5) = toc(startt);
        if verbose==1
            disp(['时间已过' num2str(conf.countedtime(i,5)) '秒'])
        end
        
        
        
        % A+ (0.5 mil)
        if flag == 1 && ~isempty(Aplus05_PPs)
            if verbose==1
                fprintf('A+ (0.5mil)  ');
            end
            conf.PPs = Aplus05_PPs;
            startt = tic;
            res{6} = scaleup_ANR(conf, low);
            conf.countedtime(i,6) = toc(startt);
            if verbose==1
                disp(['时间已过' num2str(conf.countedtime(i,6)) '秒'])
            end
        else
            res{6} = interpolated;
        end
        
        % A+ (5 mil)
        if ~isempty(Aplus_PPs)
            if verbose==1
                fprintf('A+  ');
            end
            conf.PPs = Aplus_PPs;
            startt = tic;
            res{7} = scaleup_ANR(conf, low);
            conf.countedtime(i,7) = toc(startt);
            if verbose==1
                disp(['时间已过' num2str(conf.countedtime(i,7)) '秒'])
            end
        else
            res{7} = interpolated;
        end
        
        if area_flag==1        % A+ area
            if verbose==1
                fprintf('A+ area ');
            end
            if ~isempty(Aplus_PPs_eye)
                confeye.PPs = Aplus_PPs_eye;
                startt = tic;
                eyes = get_area(interpolated, confeye.area, confeye.border*confeye.scale);
                eye_aplus = scaleup_area_ANR(confeye, eyes);
                eye_aplus = shave(eye_aplus{1}, confeye.border*confeye.scale);
                res_eye = get_area(res{7}, confeye.area);res_eye=res_eye{1};
                row_eye = get_area(img, confeye.area);row_eye=row_eye{1};
                psnr_aplus(i,1) = psnr(double(eye_aplus),double(row_eye));
                psnr_aplus(i,2) = psnr(double(res_eye),double(row_eye));
                ssim_aplus(i,1) = ssim(double(eye_aplus),double(row_eye));
                ssim_aplus(i,2) = ssim(double(res_eye),double(row_eye));
                fsim_aplus(i,1) = FeatureSIM(double(eye_aplus),double(row_eye));
                fsim_aplus(i,2) = FeatureSIM(double(res_eye),double(row_eye));
                %                 disp('--------------------------------');
                %                 disp(['psnr for a+ area_eye dict: ' num2str(psnr(double(eye_aplus),double(row_eye)))]);
                %                 disp(['psnr for a+ face dict in eye_area: ' num2str(psnr(double(res_eye),double(row_eye)))]);
                %                 disp(['psnr for a+ face dict: ' num2str(psnr( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) ))]);
                if verbose==1
                    disp(['时间已过' num2str(toc(startt)) '秒'])
                end
            end
            if ~isempty(Aplus_PPs_mouth)
                confmouth.PPs = Aplus_PPs_mouth;
                startt = tic;
                mouth = get_area(interpolated, confmouth.area, confmouth.border*confmouth.scale);
                mouth_aplus = scaleup_area_ANR(confmouth, mouth);
                mouth_aplus = shave(mouth_aplus{1}, confmouth.border*confmouth.scale);
                res_mouth = get_area(res{7}, confmouth.area);res_mouth=res_mouth{1};
                row_mouth = get_area(img, confmouth.area);row_mouth=row_mouth{1};
                psnr_aplus(i,3) = psnr(double(mouth_aplus),double(row_mouth));
                psnr_aplus(i,4) = psnr(double(res_mouth),double(row_mouth));
                psnr_aplus(i,5) = psnr( shave(double(res{7}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) );
                ssim_aplus(i,3) = ssim(double(mouth_aplus),double(row_mouth));
                ssim_aplus(i,4) = ssim(double(res_mouth),double(row_mouth));
                ssim_aplus(i,5) = ssim( shave(double(res{7}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) );
                fsim_aplus(i,3) = FeatureSIM(double(mouth_aplus),double(row_mouth));
                fsim_aplus(i,4) = FeatureSIM(double(res_mouth),double(row_mouth));
                fsim_aplus(i,5) = FeatureSIM( shave(double(res{7}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) );
           
                %                 disp(['psnr for a+ area_mouth dict: ' num2str(psnr(double(mouth_aplus),double(row_mouth)))]);
                %                 disp(['psnr for a+ face dict in mouth_area: ' num2str(psnr(double(res_mouth),double(row_mouth)))]);
                %                 disp(['psnr for a+ face dict: ' num2str(psnr( shave(double(res{5}{1}),conf.border*conf.scale) , shave(double(img{1}),conf.border*conf.scale) ))]);
                if verbose==1
                    disp(['时间已过' num2str(toc(startt)) '秒'])
                end
            end
        end
        
        %% result processing
        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1});
        
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB;
            resultCR = interpolatedCR;
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end
        %       figure;imshow(resultCB);
        %       figure;imshow(resultCR);
        
        conf.results{i} = {};
        for j = 1:numel(conf.desc)
            if j>1
                PSNRs(i,j-1) = psnr(result(:, :, j), result(:, :, 1));
                SSIMs(i,j-1) = ssim(result(:, :, j), result(:, :, 1));
            end
            
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            if ismember(j,[1,2,5,6,7,8]) || flag==1
                imwrite(result(:, :, j), conf.results{i}{j});
            end
            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
                imwrite(rgbImg, conf.resultsRGB{i}{j}); %脢脟虏脢脡芦脥录虏脜卤拢麓忙
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end
            %            figure;imshow(rgbImg);
        end
        conf.filenames{i} = f;
    end
    
    %%
    conf.duration = cputime - t;    %time of duration for upscaling step
    conf.scores.psnr = PSNRs;
    conf.scores.ssim = SSIMs;
    conf.scores.average_psnr = mean(PSNRs);
    conf.scores.average_ssim = mean(SSIMs);
    save([conf.result_dir '/' tag '_' mat_file '_results_imgscale_' num2str(imgscale)],'conf');
    save([conf.result_dir '/results_' tag '_' mat_file '_imgscale_' num2str(imgscale)],...
        'psnr_anr','psnr_aplus','ssim_anr','ssim_aplus','fsim_anr','fsim_aplus');
    
    fprintf('\n')
    disp(['PSNR&&SSIM for' conf.desc(2:8) ])
    disp( conf.scores.average_psnr)
    disp( conf.scores.average_ssim)
    % area_eye, res_eye, area_mouth, res_mouth, face
    disp(['area psnr'  ' for anr & aplus:'])
    disp( mean(psnr_anr))
    disp( mean(psnr_aplus))
    disp(['area ssim'  ' for anr & aplus:'])
    disp( mean(ssim_anr))
    disp( mean(ssim_aplus))
    disp(['area fsim'  ' for anr & aplus:'])
    disp( mean(fsim_anr))
    disp( mean(fsim_aplus))
    
end


