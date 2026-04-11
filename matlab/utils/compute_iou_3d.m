% compute_iou_3d.m
% =========================================================================
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% COMPUTE_IOU_3D  3-D IoU between two rotated bounding boxes.
%
%   iou = compute_iou_3d(box1, box2)
%
%   box: 1x7 [cx, cy, cz, length, width, height, yaw_rad]
%
%   Strategy: BEV polygon IoU x height-overlap ratio.

    % Height overlap
    z1_min = box1(3); z1_max = box1(3) + box1(6);
    z2_min = box2(3); z2_max = box2(3) + box2(6);
    h_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min));
    if h_overlap == 0, iou = 0; return; end

    % BEV polygon IoU
    c1 = bev_corners(box1);
    c2 = bev_corners(box2);
    inter_poly = sutherland_hodgman(c1, c2);
    if isempty(inter_poly), iou = 0; return; end

    inter_area = polygon_area(inter_poly);
    area1      = box1(4) * box1(5);
    area2      = box2(4) * box2(5);
    union_bev  = area1 + area2 - inter_area;
    if union_bev <= 0, iou = 0; return; end

    inter_vol = inter_area * h_overlap;
    vol1      = area1 * box1(6);
    vol2      = area2 * box2(6);
    union_vol = vol1 + vol2 - inter_vol;
    iou       = inter_vol / max(union_vol, eps);
end

function corners = bev_corners(box)
    cx = box(1); cy = box(2); hl = box(4)/2; hw = box(5)/2; yaw = box(7);
    c  = cos(yaw); s = sin(yaw);
    local = [ hl,  hw; hl, -hw; -hl, -hw; -hl,  hw];
    R     = [c, -s; s, c];
    corners = (R * local')' + [cx, cy];
end

function poly = sutherland_hodgman(subject, clip)
    output = subject;
    nc     = size(clip, 1);
    for i = 1:nc
        if isempty(output), poly = []; return; end
        a = clip(i,:); b = clip(mod(i, nc)+1,:);
        inp = output; output = []; ni = size(inp,1);
        for j = 1:ni
            cur  = inp(j,:); prev = inp(mod(j-2,ni)+1,:);
            if inside(cur,a,b)
                if ~inside(prev,a,b)
                    pt = seg_intersect(prev,cur,a,b);
                    if ~isempty(pt), output = [output; pt]; end
                end
                output = [output; cur];
            elseif inside(prev,a,b)
                pt = seg_intersect(prev,cur,a,b);
                if ~isempty(pt), output = [output; pt]; end
            end
        end
    end
    poly = output;
end

function r = inside(p,a,b)
    r = ((b(1)-a(1))*(p(2)-a(2)) - (b(2)-a(2))*(p(1)-a(1))) >= 0;
end

function pt = seg_intersect(p1,p2,p3,p4)
    d1=p2-p1; d2=p4-p3; dn=d1(1)*d2(2)-d1(2)*d2(1);
    if abs(dn)<1e-10, pt=[]; return; end
    t = ((p3(1)-p1(1))*d2(2)-(p3(2)-p1(2))*d2(1))/dn;
    pt = p1 + t*d1;
end

function area = polygon_area(poly)
    n=size(poly,1); area=0;
    for i=1:n, j=mod(i,n)+1; area=area+poly(i,1)*poly(j,2)-poly(j,1)*poly(i,2); end
    area = abs(area)/2;
end
