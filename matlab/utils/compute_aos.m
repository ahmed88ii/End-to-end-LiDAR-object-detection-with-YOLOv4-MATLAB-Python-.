function aos = compute_aos(recall, orientation_similarity)
% COMPUTE_AOS  Average Orientation Similarity (Geiger et al., CVPR 2012).
%
%   Author  : Ahmed Abdelkader
%   Project : End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
%   License : MIT
%
%   aos = compute_aos(recall, orientation_similarity)
%
%   Inputs
%   ------
%   recall                : (N,1) cumulative recall at each detection rank.
%   orientation_similarity: (N,1) cumulative mean orientation similarity
%                           score at each detection rank.
%                           Each per-detection score is
%                             s(delta_theta) = (1 + cos(delta_theta)) / 2
%                           where delta_theta = predicted_yaw - GT_yaw.
%
%   Output
%   ------
%   aos : scalar double in [0, 1].
%         1.0  =>  perfect heading prediction at every recall level.
%         0.0  =>  180-degree heading error at every recall level.
%
%   Formula
%   -------
%     AOS = (1/11) * sum_{r in {0, 0.1, ..., 1.0}}
%                     max_{r' >= r}  s(r')
%
%   Note
%   ----
%   This implementation uses the simplified orientation-similarity score
%   based on BEV-projected 2-D boxes.  Because BEV YOLO labels do not
%   contain explicit yaw annotations, delta_theta is set to 0 for all
%   matched detections (see evaluate_detector.m), so AOS == AP in
%   practice.  This is documented in the README Reproducibility section.

    narginchk(2, 2);

    validateattributes(recall, {'numeric'}, {'vector', 'real', 'finite'}, ...
                       'compute_aos', 'recall');
    validateattributes(orientation_similarity, {'numeric'}, ...
                       {'vector', 'real', 'finite'}, ...
                       'compute_aos', 'orientation_similarity');

    if numel(recall) ~= numel(orientation_similarity)
        error('compute_aos:sizeMismatch', ...
              'recall and orientation_similarity must have the same length.');
    end

    recall                 = recall(:);
    orientation_similarity = orientation_similarity(:);

    thresholds = 0 : 0.1 : 1.0;
    aos = 0;

    for t = thresholds
        mask = recall >= t;
        if any(mask)
            aos = aos + max(orientation_similarity(mask));
        end
    end

    aos = aos / numel(thresholds);
end
