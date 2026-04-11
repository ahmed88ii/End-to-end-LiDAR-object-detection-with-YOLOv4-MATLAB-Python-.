function visualize_detections(bev_image, det_boxes, det_labels, det_scores, ...
                              gt_boxes, gt_labels)
% VISUALIZE_DETECTIONS  Overlay predicted and ground-truth boxes on a BEV image.
%
%   Author  : Ahmed Abdelkader
%   Project : End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
%   License : MIT
%
%   visualize_detections(bev_image, det_boxes, det_labels, det_scores)
%   visualize_detections(bev_image, det_boxes, det_labels, det_scores, ...
%                        gt_boxes, gt_labels)
%
%   Inputs
%   ------
%   bev_image  : (H x W x 3) uint8 — Bird's-Eye-View image.
%   det_boxes  : (N x 4) double — predicted boxes in pixel coords [x,y,w,h].
%   det_labels : (N x 1) categorical or cell array of predicted class names.
%   det_scores : (N x 1) double — confidence scores in [0, 1].
%   gt_boxes   : (M x 4) double — ground-truth boxes [x,y,w,h] (optional).
%   gt_labels  : (M x 1) cell array of GT class names (optional).
%
%   Display convention
%   ------------------
%   Ground truth  : white dashed rectangles.
%   Predictions   : solid rectangles colour-coded by class:
%                     Car        -> green  [0 1 0]
%                     Pedestrian -> orange [1 0.5 0]
%                     Cyclist    -> blue   [0 0.5 1]
%                     unknown    -> red    [1 0 0]

    narginchk(4, 6);
    if nargin < 5 || isempty(gt_boxes),  gt_boxes  = []; end
    if nargin < 6 || isempty(gt_labels), gt_labels = {}; end

    figure('Name', 'BEV Detections', 'NumberTitle', 'off');
    imshow(bev_image); hold on;

    % --- Ground truth: white dashed boxes --------------------------------
    for i = 1:size(gt_boxes, 1)
        x = gt_boxes(i,1);  y = gt_boxes(i,2);
        w = gt_boxes(i,3);  h = gt_boxes(i,4);
        rectangle('Position', [x, y, w, h], ...
                  'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 1.5);
        if ~isempty(gt_labels)
            lbl = extract_label(gt_labels, i);
            text(x, max(y - 3, 5), ['GT: ' lbl], ...
                 'Color', 'w', 'FontSize', 7, 'FontWeight', 'bold');
        end
    end

    % --- Predictions: colour-coded by class ------------------------------
    colour_map  = containers.Map( ...
        {'Car', 'Pedestrian', 'Cyclist'}, ...
        {[0 1 0], [1 0.5 0], [0 0.5 1]});
    default_col = [1 0 0];

    for i = 1:size(det_boxes, 1)
        x = det_boxes(i,1);  y = det_boxes(i,2);
        w = det_boxes(i,3);  h = det_boxes(i,4);
        lbl = extract_label(det_labels, i);

        if isKey(colour_map, lbl)
            col = colour_map(lbl);
        else
            col = default_col;
        end

        rectangle('Position', [x, y, w, h], 'EdgeColor', col, 'LineWidth', 2);

        score_str = '';
        if ~isempty(det_scores) && i <= numel(det_scores)
            score_str = sprintf(' %.2f', det_scores(i));
        end
        text(x, max(y - 3, 5), [lbl score_str], ...
             'Color', col, 'FontSize', 7, 'FontWeight', 'bold');
    end

    title('BEV Object Detections  (white dashed = GT  |  solid = predictions)');
    hold off;
end


% =========================================================================
% Private helper
% =========================================================================

function s = extract_label(labels, i)
%EXTRACT_LABEL  Return the i-th label as a plain char vector.
    if iscategorical(labels)
        s = char(labels(i));
    elseif iscell(labels)
        s = labels{i};
    else
        s = char(labels(i));
    end
end
