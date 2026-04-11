% export_ground_truth.m
% =========================================================================
% Export ground-truth annotations from YOLO BEV format to a
% KITTI-inspired text format for use with third-party evaluation tools.
%
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% Input  (YOLO BEV labels)
%   data/bev/labels/<stem>.txt
%   Format: <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
%
% Output (KITTI-inspired GT files)
%   results/ground_truth/<stem>.txt
%   Format: <class> 0 0 0 <x1> <y1> <x2> <y2> 0 0 0 0 0 0 0
%
%   The field order follows the KITTI object label convention
%   (type, truncation, occlusion, alpha, 2-D box, 3-D dims, location, yaw).
%   This output is KITTI-inspired, not fully KITTI-compatible: 3-D
%   dimensions, metric location, and yaw are all placeholder zeros because
%   BEV-only YOLO labels carry no height or orientation annotations.
%
% Usage
% -----
%   run('matlab/export_ground_truth.m')
% =========================================================================

clear; clc;

% ---- Paths ---------------------------------------------------------------
LABEL_DIR  = fullfile('data', 'bev', 'labels');
IMG_DIR    = fullfile('data', 'bev', 'images');
SPLIT_FILE = fullfile('data', 'splits', 'test.txt');
OUT_DIR    = fullfile('results', 'ground_truth');

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

classNames = {'Car', 'Pedestrian', 'Cyclist'};

% ---- Load test stems -----------------------------------------------------
if exist(SPLIT_FILE, 'file')
    stems = strtrim(strsplit(fileread(SPLIT_FILE), newline));
    stems = stems(~cellfun(@isempty, stems));
else
    files = dir(fullfile(LABEL_DIR, '*.txt'));
    stems = cellfun(@(n) n(1:end-4), {files.name}, 'UniformOutput', false);
end

fprintf('Exporting GT for %d frames -> %s\n', numel(stems), OUT_DIR);

n_total = 0;
n_empty = 0;

for si = 1:numel(stems)
    stem      = stems{si};
    lbl_file  = fullfile(LABEL_DIR, [stem '.txt']);
    img_file  = fullfile(IMG_DIR,   [stem '.png']);

    % Get image dimensions to de-normalise coordinates
    if exist(img_file, 'file')
        info  = imfinfo(img_file);
        img_h = info.Height;
        img_w = info.Width;
    else
        img_h = 800;   % default from config
        img_w = 704;
    end

    out_file = fullfile(OUT_DIR, [stem '.txt']);
    fid_out  = fopen(out_file, 'w');

    if ~exist(lbl_file, 'file')
        fclose(fid_out);
        n_empty = n_empty + 1;
        continue;
    end

    fid_in = fopen(lbl_file, 'r');
    raw    = textscan(fid_in, '%d %f %f %f %f');
    fclose(fid_in);

    if isempty(raw{1})
        fclose(fid_out);
        n_empty = n_empty + 1;
        continue;
    end

    cls_ids = double(raw{1});
    cx = raw{2}; cy = raw{3}; bw = raw{4}; bh = raw{5};

    n_obj = 0;
    for i = 1:numel(cls_ids)
        ci = cls_ids(i) + 1;   % 0-indexed → 1-indexed
        if ci < 1 || ci > numel(classNames), continue; end

        x1 = (cx(i) - bw(i)/2) * img_w;
        y1 = (cy(i) - bh(i)/2) * img_h;
        x2 = (cx(i) + bw(i)/2) * img_w;
        y2 = (cy(i) + bh(i)/2) * img_h;

        % KITTI-format GT line
        % <type> trunc occ alpha x1 y1 x2 y2 h w l x y z ry
        fprintf(fid_out, '%s 0 0 0 %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0\n', ...
                classNames{ci}, x1, y1, x2, y2);
        n_obj = n_obj + 1;
    end

    fclose(fid_out);
    n_total = n_total + n_obj;
end

fprintf('  Exported %d objects across %d frames (%d empty frames).\n', ...
        n_total, numel(stems), n_empty);
fprintf('  GT files written to: %s\n', OUT_DIR);

% ---- Summary stats per class ---------------------------------------------
fprintf('\nPer-class object counts:\n');
for c = 1:numel(classNames)
    cls   = classNames{c};
    count = count_class_objects(OUT_DIR, cls);
    fprintf('  %-15s : %d\n', cls, count);
end


% =========================================================================
% Local helper
% =========================================================================

function n = count_class_objects(gt_dir, cls_name)
% COUNT_CLASS_OBJECTS  Count GT lines whose type matches cls_name.
    files = dir(fullfile(gt_dir, '*.txt'));
    n = 0;
    for fi = 1:numel(files)
        fid  = fopen(fullfile(files(fi).folder, files(fi).name), 'r');
        data = textscan(fid, '%s %*[^\n]');
        fclose(fid);
        if isempty(data{1}), continue; end
        n = n + sum(strcmp(data{1}, cls_name));
    end
end
