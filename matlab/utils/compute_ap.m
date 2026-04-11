function ap = compute_ap(recall, precision, method)
% COMPUTE_AP  Average Precision via 11-point or area-under-curve interpolation.
%
%   Author  : Ahmed Abdelkader
%   Project : End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
%   License : MIT
%
%   ap = compute_ap(recall, precision)
%   ap = compute_ap(recall, precision, method)
%
%   Inputs
%   ------
%   recall    : (N,1) or (1,N) numeric — cumulative recall values in [0,1],
%               sorted in ascending order.
%   precision : (N,1) or (1,N) numeric — precision at each recall level.
%   method    : char — '11point' (PASCAL VOC, default) | 'auc' (trapezoidal)
%
%   Output
%   ------
%   ap        : scalar double in [0, 1]
%
%   Notes
%   -----
%   '11point' : 11-point interpolation used by PASCAL VOC 2010 and earlier.
%               Evaluates max precision at recall levels {0, 0.1, ..., 1.0}.
%   'auc'     : Strictly monotonically-decreasing envelope of the PR curve,
%               integrated with the trapezoidal rule.

    narginchk(2, 3);
    if nargin < 3 || isempty(method)
        method = '11point';
    end

    validateattributes(recall,    {'numeric'}, {'vector', 'real', 'finite'}, ...
                       'compute_ap', 'recall');
    validateattributes(precision, {'numeric'}, {'vector', 'real', 'finite'}, ...
                       'compute_ap', 'precision');

    recall    = recall(:);
    precision = precision(:);

    if numel(recall) ~= numel(precision)
        error('compute_ap:sizeMismatch', ...
              'recall and precision must have the same number of elements.');
    end

    switch lower(method)

        case '11point'
            % PASCAL VOC 11-point interpolation
            thresholds = 0 : 0.1 : 1.0;
            ap = 0;
            for t = thresholds
                mask = recall >= t;
                if any(mask)
                    ap = ap + max(precision(mask));
                end
            end
            ap = ap / numel(thresholds);

        case 'auc'
            % Area under the monotone PR envelope (trapezoidal rule).
            % Prepend sentinel (recall=0, precision=1) and
            % append sentinel (recall=1, precision=0).
            r = [0; recall; 1];
            p = [1; precision; 0];

            % Build monotonically non-increasing precision envelope
            % (running maximum from right to left).
            for i = numel(p) - 1 : -1 : 1
                p(i) = max(p(i), p(i + 1));
            end

            % Trapezoidal integration
            ap = sum(diff(r) .* p(2:end));

        otherwise
            error('compute_ap:unknownMethod', ...
                  'Unknown method "%s". Use ''11point'' or ''auc''.', method);
    end
end
