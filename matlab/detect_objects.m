% detect_objects.m
% =========================================================================
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% =========================================================================
% Run inference with a trained YOLOv4 detector on BEV images.
%
% Set MODEL_FILE and INPUT, then run.
% Writes per-image detection .txt files to OUTPUT_DIR.
% Format: <class> <score> <x1> <y1> <x2> <y2>
% =========================================================================

clear; clc;
addpath(fullfile('matlab', 'utils'));

MODEL_FILE  = fullfile('matlab', 'trained_models', 'yolov4_bev_latest.mat');
INPUT       = fullfile('data', 'bev', 'images');
OUTPUT_DIR  = fullfile('matlab', 'detections');
CONF_THRESH = 0.30;
DISPLAY     = false;

if ~exist(OUTPUT_DIR,'dir'), mkdir(OUTPUT_DIR); end

if ~exist(MODEL_FILE,'file')
    error('Model not found: %s\nRun train_yolov4.m first.', MODEL_FILE);
end
fprintf('Loading %s ...\n', MODEL_FILE);
M = load(MODEL_FILE, 'detector', 'classNames');
detector = M.detector; classNames = M.classNames;

% Collect images
if isfolder(INPUT)
    files     = dir(fullfile(INPUT,'*.png'));
    img_paths = fullfile({files.folder},{files.name});
else
    img_paths = {INPUT};
end
fprintf('Detecting in %d images...\n', numel(img_paths));

total = 0; t0 = tic;
for i = 1:numel(img_paths)
    ip = img_paths{i};
    if ~exist(ip,'file'), warning('Not found: %s', ip); continue; end
    img = imread(ip);

    [h,w,~] = size(img);
    det_sz  = detector.InputSize(1:2);
    if ~isequal([h,w], det_sz)
        img_in = imresize(img, det_sz);
        sx = w/det_sz(2); sy = h/det_sz(1);
    else
        img_in = img; sx = 1; sy = 1;
    end

    [bboxes, scores, labels] = detect(detector, img_in, ...
        'Threshold', CONF_THRESH, 'SelectStrongest', true, 'MiniBatchSize', 1);

    if ~isempty(bboxes)
        bboxes(:,1)=bboxes(:,1)*sx; bboxes(:,2)=bboxes(:,2)*sy;
        bboxes(:,3)=bboxes(:,3)*sx; bboxes(:,4)=bboxes(:,4)*sy;
    end

    [~,stem,~] = fileparts(ip);
    fid = fopen(fullfile(OUTPUT_DIR,[stem '.txt']),'w');
    for d = 1:size(bboxes,1)
        fprintf(fid,'%s %.4f %.1f %.1f %.1f %.1f\n', ...
                char(labels(d)), scores(d), ...
                bboxes(d,1), bboxes(d,2), ...
                bboxes(d,1)+bboxes(d,3), bboxes(d,2)+bboxes(d,4));
    end
    fclose(fid);
    total = total + size(bboxes,1);

    if DISPLAY && ~isempty(bboxes)
        visualize_detections(img, bboxes, labels, scores);
        title(stem); drawnow;
    end
    if mod(i,50)==0, fprintf('  %d / %d\n', i, numel(img_paths)); end
end

fprintf('\nDone: %d images, %d detections, %.2fs (%.1f img/s)\n', ...
        numel(img_paths), total, toc(t0), numel(img_paths)/toc(t0));
