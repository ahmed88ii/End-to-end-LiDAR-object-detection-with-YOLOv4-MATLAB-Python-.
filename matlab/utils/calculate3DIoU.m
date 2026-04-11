function iou = calculate3DIoU(box1, box2)
% CALCULATE3DIOU  Full 3-D IoU between two rotated bounding boxes.
%
%   Author : Ahmed Abdelkader
%   Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
%   License: MIT
%
%   iou = calculate3DIoU(box1, box2)
%
%   Each input is a 1-by-7 vector:
%       [cx, cy, cz, length, width, height, yaw_rad]
%
%       cx, cy, cz  : centre coordinates (metres)
%       length      : extent along the local x-axis
%       width       : extent along the local y-axis
%       height      : extent along the local z-axis
%       yaw_rad     : rotation about the z-axis (radians)
%
%   Returns scalar iou in [0, 1].
%
%   Algorithm
%   ---------
%   IoU_3D = (intersection volume) / (union volume)
%
%   Decomposed as:
%       intersection volume = (BEV polygon intersection area) * (height overlap)
%       union volume        = vol1 + vol2 - intersection volume
%
%   The BEV polygon intersection is computed via the Sutherland–Hodgman
%   algorithm on the four rotated corners of each box.
%
%   See also: compute_iou_3d (legacy alias kept for backward compatibility)

    % ---- Validate inputs --------------------------------------------------
    if numel(box1) ~= 7 || numel(box2) ~= 7
        error('calculate3DIoU:badInput', ...
              'Each box must be a 1-by-7 vector [cx,cy,cz,l,w,h,yaw].');
    end
    box1 = box1(:)';
    box2 = box2(:)';

    % ---- Height (z) overlap -----------------------------------------------
    % Treat z as the bottom face of the box (bottom = cz, top = cz + height)
    z1_bot = box1(3);  z1_top = box1(3) + box1(6);
    z2_bot = box2(3);  z2_top = box2(3) + box2(6);

    h_overlap = max(0, min(z1_top, z2_top) - max(z1_bot, z2_bot));

    if h_overlap <= 0
        iou = 0;
        return;
    end

    % ---- BEV polygon IoU --------------------------------------------------
    corners1 = bev_corners(box1);   % 4-by-2
    corners2 = bev_corners(box2);   % 4-by-2

    inter_poly = sutherland_hodgman(corners1, corners2);
    if isempty(inter_poly)
        iou = 0;
        return;
    end

    inter_area = polygon_area(inter_poly);
    area1      = box1(4) * box1(5);
    area2      = box2(4) * box2(5);
    union_bev  = area1 + area2 - inter_area;

    if union_bev <= 0
        iou = 0;
        return;
    end

    % ---- Volumetric IoU ---------------------------------------------------
    inter_vol = inter_area * h_overlap;
    vol1      = area1 * box1(6);
    vol2      = area2 * box2(6);
    union_vol = vol1 + vol2 - inter_vol;

    iou = inter_vol / max(union_vol, eps);
end


% =========================================================================
% Private helpers
% =========================================================================

function corners = bev_corners(box)
% BEV_CORNERS  Compute the 4 rotated corners of a box in the XY plane (4-by-2).
    cx  = box(1);   cy  = box(2);
    hl  = box(4) / 2;   hw = box(5) / 2;
    yaw = box(7);

    c = cos(yaw);  s = sin(yaw);
    R = [c, -s; s, c];

    local = [ hl,  hw;
              hl, -hw;
             -hl, -hw;
             -hl,  hw];

    corners = (R * local')' + [cx, cy];
end


function poly = sutherland_hodgman(subject, clip)
% SUTHERLAND_HODGMAN  Clip a convex polygon against another convex polygon.
%
%   subject, clip : N-by-2 vertex arrays (convex, CCW or CW consistent).
%   Returns the intersection polygon vertices, or [] if no intersection.

    output = subject;
    n_clip = size(clip, 1);

    for i = 1:n_clip
        if isempty(output)
            poly = [];
            return;
        end

        a = clip(i, :);
        b = clip(mod(i, n_clip) + 1, :);

        input_pts = output;
        output    = [];
        n_in      = size(input_pts, 1);

        for j = 1:n_in
            cur  = input_pts(j, :);
            prev = input_pts(mod(j - 2, n_in) + 1, :);

            if inside_edge(cur, a, b)
                if ~inside_edge(prev, a, b)
                    pt = edge_intersect(prev, cur, a, b);
                    if ~isempty(pt)
                        output = [output; pt];  %#ok<AGROW>
                    end
                end
                output = [output; cur];   %#ok<AGROW>
            elseif inside_edge(prev, a, b)
                pt = edge_intersect(prev, cur, a, b);
                if ~isempty(pt)
                    output = [output; pt];  %#ok<AGROW>
                end
            end
        end
    end
    poly = output;
end


function flag = inside_edge(p, a, b)
% INSIDE_EDGE  True if point p is on the left side (or on) edge a→b.
    flag = ((b(1) - a(1)) * (p(2) - a(2)) - (b(2) - a(2)) * (p(1) - a(1))) >= 0;
end


function pt = edge_intersect(p1, p2, p3, p4)
% EDGE_INTERSECT  Parametric intersection of segment p1-p2 with line p3-p4.
    d1    = p2 - p1;
    d2    = p4 - p3;
    denom = d1(1) * d2(2) - d1(2) * d2(1);
    if abs(denom) < 1e-10
        pt = [];
        return;
    end
    t  = ((p3(1) - p1(1)) * d2(2) - (p3(2) - p1(2)) * d2(1)) / denom;
    pt = p1 + t * d1;
end


function area = polygon_area(poly)
% POLYGON_AREA  Signed area via shoelace formula; returns absolute value.
    n    = size(poly, 1);
    area = 0;
    for i = 1:n
        j    = mod(i, n) + 1;
        area = area + poly(i,1) * poly(j,2) - poly(j,1) * poly(i,2);
    end
    area = abs(area) / 2;
end
