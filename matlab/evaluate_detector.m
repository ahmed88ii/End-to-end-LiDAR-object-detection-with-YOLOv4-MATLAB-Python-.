% evaluate_detector.m
% =========================================================================
% Evaluate a trained Complex-YOLOv4 detector and report per-class metrics.
%
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% Metrics computed
% ----------------
%   - Precision and Recall at each confidence threshold
%   - Average Precision (AP)  via 11-point PASCAL VOC interpolation
%   - Average Orientation Similarity (AOS, Geiger et al. CVPR 2012)
%   - IoU distribution histogram per class
%
% IoU thresholds used  (matching KITTI convention)
%   Car        : 0.7
%   Pedestrian : 0.5
%   Cyclist    : 0.5
%
% Input files
% -----------
%   Detection files : matlab/detections/<stem>.txt
%       Format: <class> <score> <x1> <y1> <x2> <y2>  (pixel coords)
%   Ground truth    : data/bev/labels/<stem>.txt
%       Format: <class_id> <cx_n> <cy_n> <w_n> <h_n>  (YOLO normalised)
%
% Usage
% -----
%   Run after detect_objects.m has written detection files.
%   Results are printed to the console and saved as plots in results/.
% =========================================================================

clear; clc;
addpath(fullfile('matlab', 'utils'));

% ---- Paths and settings --------------------------------------------------
DET_DIR    = fullfile('matlab', 'detections');
GT_DIR     = fullfile('data',   'bev', 'labels');
IMG_DIR    = fullfile('data',   'bev', 'images');
SPLIT_FILE = fullfile('data',   'splits', 'test.txt');
RESULTS_DIR = fullfile('results');

if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

classNames  = {'Car', 'Pedestrian', 'Cyclist'};
iou_per_cls = containers.Map({'Car','Pedestrian','Cyclist'}, {0.7, 0.5, 0.5});

% ---- Load test frame stems -----------------------------------------------
if exist(SPLIT_FILE, 'file')
    stems = strtrim(strsplit(fileread(SPLIT_FILE), newline));
    stems = stems(~cellfun(@isempty, stems));
else
    files = dir(fullfile(DET_DIR, '*.txt'));
    stems = cellfun(@(n) n(1:end-4), {files.name}, 'UniformOutput', false);
end
fprintf('Evaluating on %d test frames ...\n\n', numel(stems));

% ---- Accumulation buffers ------------------------------------------------
% all_det.(class) = [score, tp, fp, iou_best, delta_yaw]
% n_gt.(class)    = total GT count
for c = 1:numel(classNames)
    cls = classNames{c};
    all_det.(cls) = zeros(0, 5);
    n_gt.(cls)    = 0;
end

% ---- Main loop: match detections to GT per frame -------------------------
for si = 1:numel(stems)
    stem     = stems{si};
    det_file = fullfile(DET_DIR, [stem '.txt']);
    gt_file  = fullfile(GT_DIR,  [stem '.txt']);
    img_file = fullfile(IMG_DIR, [stem '.png']);

    if ~exist(img_file, 'file'), continue; end
    info  = imfinfo(img_file);
    img_h = info.Height;
    img_w = info.Width;

    % --- Parse ground truth ---
    gt_boxes = struct();
    for c = 1:numel(classNames)
        gt_boxes.(classNames{c}) = zeros(0, 5);   % [x, y, w, h, yaw]
    end

    if exist(gt_file, 'file')
        fid = fopen(gt_file, 'r');
        raw = textscan(fid, '%d %f %f %f %f');
        fclose(fid);
        if ~isempty(raw{1})
            cls_ids = double(raw{1});
            cx = raw{2}; cy = raw{3}; bw = raw{4}; bh = raw{5};
            for i = 1:numel(cls_ids)
                ci = cls_ids(i) + 1;
                if ci < 1 || ci > numel(classNames), continue; end
                x = (cx(i) - bw(i)/2) * img_w;
                y = (cy(i) - bh(i)/2) * img_h;
                w = bw(i) * img_w;
                h = bh(i) * img_h;
                gt_boxes.(classNames{ci}) = ...
                    [gt_boxes.(classNames{ci}); x, y, w, h, 0];   % yaw=0 (BEV)
                n_gt.(classNames{ci}) = n_gt.(classNames{ci}) + 1;
            end
        end
    end

    % --- Parse detections ---
    if ~exist(det_file, 'file'), continue; end
    fid = fopen(det_file, 'r');
    raw = textscan(fid, '%s %f %f %f %f %f');
    fclose(fid);
    if isempty(raw{1}), continue; end

    det_cls    = raw{1};
    det_scores = raw{2};
    det_x1 = raw{3}; det_y1 = raw{4};
    det_x2 = raw{5}; det_y2 = raw{6};
    det_w  = det_x2 - det_x1;
    det_h  = det_y2 - det_y1;

    for c = 1:numel(classNames)
        cls      = classNames{c};
        mask     = strcmp(det_cls, cls);
        if ~any(mask), continue; end

        d_boxes  = [det_x1(mask), det_y1(mask), det_w(mask), det_h(mask)];
        d_scores = det_scores(mask);
        g_boxes  = gt_boxes.(cls);
        iou_thr  = iou_per_cls(cls);

        n_det = size(d_boxes, 1);
        n_g   = size(g_boxes, 1);

        tp       = zeros(n_det, 1);
        fp       = zeros(n_det, 1);
        iou_best = zeros(n_det, 1);
        d_yaw    = zeros(n_det, 1);   % orientation error (rad)

        [~, order] = sort(d_scores, 'descend');
        matched    = false(n_g, 1);

        for di = 1:n_det
            idx   = order(di);
            b_iou = 0;
            b_g   = -1;

            for gi = 1:n_g
                if matched(gi), continue; end
                iou_val = bev_iou_2d(d_boxes(idx,:), g_boxes(gi,1:4));
                if iou_val > b_iou
                    b_iou = iou_val;
                    b_g   = gi;
                end
            end

            iou_best(idx) = b_iou;
            if b_iou >= iou_thr && b_g > 0
                tp(idx)    = 1;
                matched(b_g) = true;
                % Orientation error: assume yaw stored in col 5; 0 if unavailable
                if size(g_boxes, 2) >= 5
                    d_yaw(idx) = 0;   % BEV labels have no yaw — set to 0
                end
            else
                fp(idx) = 1;
            end
        end

        os = (1 + cos(d_yaw)) / 2;   % orientation similarity per detection
        all_det.(cls) = [all_det.(cls); [d_scores, tp, fp, iou_best, os]];
    end
end

% ---- Compute and display metrics per class --------------------------------
fprintf('%-15s  %8s  %8s  %8s  %8s\n', 'Class', 'AP', 'AOS', 'Prec', 'Recall');
fprintf('%s\n', repmat('-', 1, 55));

results = struct();

for c = 1:numel(classNames)
    cls  = classNames{c};
    data = sortrows(all_det.(cls), -1);   % sort by confidence descending

    if isempty(data) || n_gt.(cls) == 0
        fprintf('%-15s  %8.4f  %8.4f  %8.4f  %8.4f\n', cls, 0, 0, 0, 0);
        results.(cls) = struct('AP',0,'AOS',0,'Precision',[],'Recall',[]);
        continue;
    end

    tp_cum    = cumsum(data(:, 2));
    fp_cum    = cumsum(data(:, 3));
    recall    = tp_cum ./ n_gt.(cls);
    precision = tp_cum ./ (tp_cum + fp_cum + eps);

    % Cumulative orientation similarity score
    os_cum    = cumsum(data(:, 5));
    os_score  = os_cum ./ (tp_cum + fp_cum + eps);

    ap  = compute_ap(recall, precision, '11point');
    aos = compute_aos(recall, os_score);

    % Final precision/recall (at default 0.5 confidence cutoff)
    conf_mask = data(:, 1) >= 0.5;
    if any(conf_mask)
        final_prec   = mean(precision(conf_mask));
        final_recall = max(recall(conf_mask));
    else
        final_prec   = precision(end);
        final_recall = recall(end);
    end

    fprintf('%-15s  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
            cls, ap, aos, final_prec, final_recall);

    results.(cls) = struct('AP', ap, 'AOS', aos, ...
                           'Precision', precision, 'Recall', recall, ...
                           'IoU', data(:, 4));
end

fprintf('\nEvaluation complete.\n\n');

% ---- Plot precision-recall curves ----------------------------------------
fig_pr = figure('Name', 'Precision-Recall Curves', 'NumberTitle', 'off');
colors = {'b', 'r', 'g'};

for c = 1:numel(classNames)
    cls = classNames{c};
    if ~isfield(results, cls) || isempty(results.(cls).Recall), continue; end
    plot(results.(cls).Recall, results.(cls).Precision, ...
         colors{c}, 'LineWidth', 1.5, ...
         'DisplayName', sprintf('%s (AP=%.3f)', cls, results.(cls).AP));
    hold on;
end
xlabel('Recall'); ylabel('Precision');
title('Precision-Recall Curves (BEV, 11-point AP)');
legend('Location', 'northeast'); grid on; hold off;
xlim([0 1]); ylim([0 1]);
saveas(fig_pr, fullfile(RESULTS_DIR, 'precision_recall_curves.png'));

% ---- Plot IoU distribution histograms ------------------------------------
fig_iou = figure('Name', 'IoU Distribution', 'NumberTitle', 'off');
for c = 1:numel(classNames)
    cls = classNames{c};
    subplot(1, numel(classNames), c);
    if isfield(results, cls) && ~isempty(results.(cls).IoU)
        iou_vals = results.(cls).IoU;
        iou_vals = iou_vals(iou_vals > 0);   % only matched detections
        if ~isempty(iou_vals)
            histogram(iou_vals, 0:0.05:1, 'FaceColor', colors{c});
            xline(iou_per_cls(cls), 'k--', 'LineWidth', 1.5, ...
                  'DisplayName', sprintf('Threshold=%.1f', iou_per_cls(cls)));
        end
    end
    xlabel('IoU'); ylabel('Count');
    title(cls); grid on;
    xlim([0 1]);
end
sgtitle('IoU Distribution (matched detections)');
saveas(fig_iou, fullfile(RESULTS_DIR, 'iou_distribution.png'));

fprintf('Plots saved to %s\n', RESULTS_DIR);

% ---- Save numeric results ------------------------------------------------
results_file = fullfile(RESULTS_DIR, 'evaluation_results.mat');
save(results_file, 'results', 'classNames', 'n_gt');
fprintf('Results saved to %s\n', results_file);


% =========================================================================
% Local helper
% =========================================================================

function iou = bev_iou_2d(b1, b2)
% BEV_IOU_2D  Axis-aligned 2D IoU for [x, y, w, h] boxes.
    x1    = max(b1(1), b2(1));   y1 = max(b1(2), b2(2));
    x2    = min(b1(1)+b1(3), b2(1)+b2(3));
    y2    = min(b1(2)+b1(4), b2(2)+b2(4));
    inter = max(0, x2-x1) * max(0, y2-y1);
    union = b1(3)*b1(4) + b2(3)*b2(4) - inter;
    iou   = inter / max(union, eps);
end
