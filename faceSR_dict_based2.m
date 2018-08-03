% Anchored Neighborhood Regression for Face Super-Resolution
% using Bicubic, GR, ANR, A+(0.5 mil), A+.
% version2.0 add locality constrain


%clc
%clear;
addpath('./utilities');
verbose = 0;

%       % make sure you have make ompbox and ksvdbox already,
%       % if not run the code, and before make you should have a gcc compiler
% addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
% addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm

imgscale = 1; % the scale reference we work with
flag = 1;       % flag = 0 - only GR, ANR, A+, and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied

upscaling = 2; % the magnification factor x2, x3, x4...
% train_dir = 'CMDP_train';
train_dir = 'train_faces';
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
	tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];

%     mat_file = ['conf_Zeyde_CMDP' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];
	mat_file = ['conf_Zeyde_' train_dir '_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];

    %% load or learn dict
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file '.mat']);
        load(mat_file, 'conf');
    else
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' for ' train_dir ' dataset using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform
        conf.window = [4 4]; % low-res. window size
        conf.border = [1 1]; % border of the lr image (to ignore) %in sampling-grid.m will mutiply with scale

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
  t = cputime;
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

    
    fprintf('\n')
    disp(['PSNR&&SSIM for' conf.desc(2:8) ])
    disp( conf.scores.average_psnr)
    disp( conf.scores.average_ssim)


end


