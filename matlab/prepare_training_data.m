% prepare_training_data.m
% =========================================================================
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% =========================================================================
% Build MATLAB datastores from BEV images + YOLO labels, estimate anchors.
%
% Label format (YOLO .txt, one per image, same stem):
%   <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>  (all in [0,1])
%
% Outputs (written to workspace and saved to disk):
%   trainData, valData  -- combined image+box datastores
%   classNames          -- string array
%   anchorBoxes         -- cell array for 3 YOLOv4 prediction heads
% =========================================================================

clear; clc;

IMAGE_DIR    = fullfile('data', 'bev', 'images');
LABEL_DIR    = fullfile('data', 'bev', 'labels');
SPLIT_DIR    = fullfile('data', 'splits');
ANCHOR_FILE  = fullfile('matlab', 'checkpoints', 'anchor_boxes.mat');

if ~exist(fileparts(ANCHOR_FILE), 'dir')
    mkdir(fileparts(ANCHOR_FILE));
end

classNames = ["Car", "Pedestrian", "Cyclist"];

% ---- Load a split file (returns cell array of stems) --------------------
function stems = load_split(split_dir, name)
    fpath = fullfile(split_dir, [name '.txt']);
    if ~exist(fpath,'file'), stems={}; return; end
    raw   = strtrim(strsplit(fileread(fpath), newline));
    stems = raw(~cellfun(@isempty, raw));
end

% ---- Parse one YOLO label file -> table row -----------------------------
function row = parse_label(lbl_path, img_h, img_w, class_names)
    nc  = numel(class_names);
    bxs = cell(1, nc);
    for k=1:nc, bxs{k}=zeros(0,4); end
    if ~exist(lbl_path,'file')
        row = cell2table(bxs,'VariableNames',cellstr(class_names)); return;
    end
    fid = fopen(lbl_path,'r');
    D   = textscan(fid,'%d %f %f %f %f'); fclose(fid);
    if isempty(D{1})
        row = cell2table(bxs,'VariableNames',cellstr(class_names)); return;
    end
    for i = 1:numel(D{1})
        c = D{1}(i)+1; if c<1||c>nc, continue; end
        x = (D{2}(i)-D{4}(i)/2)*img_w; y = (D{3}(i)-D{5}(i)/2)*img_h;
        w = D{4}(i)*img_w;              h = D{5}(i)*img_h;
        bxs{c} = [bxs{c}; x y w h];
    end
    row = cell2table(bxs,'VariableNames',cellstr(class_names));
end

% ---- Build table --------------------------------------------------------
function tbl = build_table(stems, img_dir, lbl_dir, class_names)
    n     = numel(stems);
    paths = cell(n,1);
    rows  = cell(n, numel(class_names));
    for i = 1:n
        ip = fullfile(img_dir, [stems{i} '.png']);
        lp = fullfile(lbl_dir, [stems{i} '.txt']);
        paths{i} = ip;
        info = imfinfo(ip);
        r    = parse_label(lp, info.Height, info.Width, class_names);
        for c=1:numel(class_names), rows{i,c}=r{:,c}; end
    end
    tbl = [cell2table(paths,'VariableNames',{'imageFilename'}), ...
           array2table(rows,'VariableNames',cellstr(class_names))];
end

train_stems = load_split(SPLIT_DIR, 'train');
val_stems   = load_split(SPLIT_DIR, 'val');
fprintf('Train: %d  |  Val: %d\n', numel(train_stems), numel(val_stems));

fprintf('Loading labels...\n');
trainTable = build_table(train_stems, IMAGE_DIR, LABEL_DIR, classNames);
valTable   = build_table(val_stems,   IMAGE_DIR, LABEL_DIR, classNames);

trainData = combine(imageDatastore(trainTable.imageFilename), ...
                    boxLabelDatastore(trainTable(:, cellstr(classNames))));
valData   = combine(imageDatastore(valTable.imageFilename), ...
                    boxLabelDatastore(valTable(:, cellstr(classNames))));

fprintf('Estimating anchor boxes (k=6)...\n');
[allAnchors, meanIoU] = estimateAnchorBoxes(trainData, 6);
fprintf('  Mean IoU: %.4f\n', meanIoU);

[~, idx]    = sort(prod(allAnchors, 2));
sorted      = allAnchors(idx, :);
anchorBoxes = {sorted(1:2,:); sorted(3:4,:); sorted(5:6,:)};

save(ANCHOR_FILE, 'anchorBoxes', 'meanIoU');
fprintf('Anchors saved to %s\nData preparation complete.\n', ANCHOR_FILE);
