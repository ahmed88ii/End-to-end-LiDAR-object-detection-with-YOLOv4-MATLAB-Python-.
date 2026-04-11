% generate_bev.m
% =========================================================================
% Generate Bird's-Eye-View (BEV) images from PCD point clouds.
%
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% Each output is a 3-channel uint8 PNG where the channels encode:
%   Channel 1 (R) - Height map   : maximum z-value in each voxel column,
%                                  normalised by MAX_HEIGHT → [0, 255]
%   Channel 2 (G) - Intensity map: mean reflectivity of all points in the
%                                  voxel column, in [0, 1] → [0, 255]
%   Channel 3 (B) - Density map  : point count per voxel column, clamped
%                                  at MAX_DENSITY and scaled → [0, 255]
%
% Spatial extent (defaults match the thesis configuration):
%   X (forward)  :  0 → 70.4 m   →  704 pixels  (0.1 m / pixel)
%   Y (lateral)  : -40 → 40 m    →  800 pixels  (0.1 m / pixel)
%   Z (height)   : -3 → 1 m      (only points in this band are used)
%
% Usage
% -----
%   % Default: process all PCD files and write PNGs
%   generate_bev
%
%   % Override paths at the top of the script or call as a function from
%   % another script after setting the PCD_DIR / OUT_DIR variables.
% =========================================================================

clear; clc;

% ---- Configuration -------------------------------------------------------
PCD_DIR     = fullfile('data', 'pcd');
OUT_DIR     = fullfile('data', 'bev', 'images');

% BEV spatial extent
X_MIN = 0;      X_MAX = 70.4;   % metres
Y_MIN = -40;    Y_MAX = 40;
Z_MIN = -3;     Z_MAX = 1;

RESOLUTION   = 0.1;             % metres per pixel
MAX_HEIGHT   = Z_MAX - Z_MIN;   % normalisation ceiling
MAX_DENSITY  = 64;              % point-count ceiling for density channel

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

% Derived grid dimensions
BEV_W = round((X_MAX - X_MIN) / RESOLUTION);   % 704
BEV_H = round((Y_MAX - Y_MIN) / RESOLUTION);   % 800

% ---- Collect PCD files ---------------------------------------------------
files = dir(fullfile(PCD_DIR, '*.pcd'));
if isempty(files)
    error('No .pcd files found in %s\nRun lvx_to_pcd.py first.', PCD_DIR);
end
fprintf('Generating BEV images for %d point cloud(s) ...\n', numel(files));

% ---- Process each file ---------------------------------------------------
for fi = 1:numel(files)
    pcd_path = fullfile(files(fi).folder, files(fi).name);
    [~, stem, ~] = fileparts(files(fi).name);

    pts = load_pcd(pcd_path);   % Nx4 [x, y, z, intensity]
    if isempty(pts)
        warning('Empty point cloud: %s', files(fi).name);
        continue;
    end

    bev = points_to_bev(pts, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, ...
                        RESOLUTION, BEV_W, BEV_H, MAX_HEIGHT, MAX_DENSITY);

    out_path = fullfile(OUT_DIR, [stem '.png']);
    imwrite(bev, out_path);

    if mod(fi, 50) == 0 || fi == numel(files)
        fprintf('  [%d / %d] %s\n', fi, numel(files), files(fi).name);
    end
end

fprintf('BEV images written to: %s\n', OUT_DIR);


% =========================================================================
% Local functions
% =========================================================================

function pts = load_pcd(pcd_path)
% LOAD_PCD  Read an ASCII .pcd file and return Nx4 [x, y, z, intensity].
%
%   Supports the PCL ASCII format written by lvx_to_pcd.py.
%   Returns an empty 0x4 matrix on error.

    pts = zeros(0, 4, 'single');
    fid = fopen(pcd_path, 'r');
    if fid < 0
        warning('Cannot open: %s', pcd_path);
        return;
    end

    % Skip header lines until "DATA ascii"
    data_started = false;
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if startsWith(upper(line), 'DATA')
            data_started = true;
            break;
        end
    end

    if ~data_started
        fclose(fid);
        warning('No DATA section in: %s', pcd_path);
        return;
    end

    raw = textscan(fid, '%f %f %f %f');
    fclose(fid);

    if isempty(raw{1}), return; end
    pts = single([raw{1}, raw{2}, raw{3}, raw{4}]);
end


function bev = points_to_bev(pts, x_min, x_max, y_min, y_max, z_min, z_max, ...
                              res, bev_w, bev_h, max_h, max_dens)
% POINTS_TO_BEV  Project a point cloud onto a 3-channel BEV image.
%
%   pts     : Nx4 single [x, y, z, intensity]
%   Returns : (bev_h x bev_w x 3) uint8 image

    % ----- Crop to ROI -----
    mask = pts(:,1) >= x_min & pts(:,1) < x_max & ...
           pts(:,2) >= y_min & pts(:,2) < y_max & ...
           pts(:,3) >= z_min & pts(:,3) <= z_max;
    pts  = pts(mask, :);

    % ----- Pixel indices -----
    % X (forward) → column; Y (lateral) → row (inverted so +Y is left on image)
    col = floor((pts(:,1) - x_min) / res) + 1;                    % 1..bev_w
    row = floor((pts(:,2) - y_min) / res) + 1;                    % 1..bev_h
    row = bev_h - row + 1;                                         % flip Y

    % Clamp to valid range
    col = max(1, min(bev_w, col));
    row = max(1, min(bev_h, row));

    % Linear indices
    lin = (col - 1) * bev_h + row;    % column-major

    % ----- Accumulators -----
    height_map    = zeros(bev_h, bev_w, 'single');
    intensity_sum = zeros(bev_h, bev_w, 'single');
    density_map   = zeros(bev_h, bev_w, 'single');

    z_norm = single((pts(:,3) - z_min) / max(max_h, eps));   % [0, 1]

    % Height: keep maximum normalised z per cell
    for i = 1:numel(lin)
        r = row(i); c = col(i);
        if z_norm(i) > height_map(r, c)
            height_map(r, c) = z_norm(i);
        end
        intensity_sum(r, c) = intensity_sum(r, c) + pts(i, 4);
        density_map(r, c)   = density_map(r, c)   + 1;
    end

    % Intensity: mean per cell
    occupied = density_map > 0;
    intensity_map = zeros(bev_h, bev_w, 'single');
    intensity_map(occupied) = intensity_sum(occupied) ./ density_map(occupied);

    % Density: clamp to [0, 1]
    density_norm = min(density_map / max_dens, 1);

    % ----- Pack into uint8 RGB -----
    bev = uint8(cat(3, ...
        height_map    * 255, ...   % R: height
        intensity_map * 255, ...   % G: intensity
        density_norm  * 255));     % B: density
end
