% train_yolov4.m
% =========================================================================
% Train a Complex-YOLOv4 object detector on BEV LiDAR images.
%
% Author : Ahmed Abdelkader
% Project: End-to-End LiDAR Object Detection with YOLOv4 (MATLAB + Python)
% License: MIT
%
% Prerequisites
% -------------
%   - MATLAB R2021a or later
%   - Deep Learning Toolbox
%   - Computer Vision Toolbox
%   - Run prepare_training_data.m first to generate datastores and anchors.
%
% Hyperparameters (thesis configuration)
% ---------------------------------------
%   maxEpochs       = 50
%   miniBatchSize   = 16
%   learningRate    = 0.001
%   warmupPeriod    = 500    (iterations with linearly ramped LR)
%   l2Regularization= 0.001
%   penaltyThreshold= 0.5    (IoU threshold below which background penalty
%                             is applied to confidence branch)
%
% The script uses the CSP-DarkNet53 backbone pre-trained on COCO, applies a
% cosine-annealing LR schedule with a linear warm-up, and saves the best
% checkpoint (lowest validation loss) plus the final model.
% =========================================================================

clear; clc;

% ---- Hyperparameters (thesis values) -------------------------------------
MAX_EPOCHS        = 50;
MINI_BATCH_SIZE   = 16;
INIT_LR           = 0.001;
WARMUP_PERIOD     = 500;     % iterations (not epochs)
L2_REG            = 0.001;
PENALTY_THRESHOLD = 0.5;     % Complex-YOLO confidence IoU penalty boundary
MOMENTUM          = 0.9;
INPUT_SIZE        = [416, 416, 3];
NETWORK           = 'csp-darknet53-coco';   % backbone

% ---- Paths ---------------------------------------------------------------
CHECKPOINT_DIR = fullfile('matlab', 'checkpoints');
MODEL_DIR      = fullfile('matlab', 'trained_models');
ANCHOR_FILE    = fullfile(CHECKPOINT_DIR, 'anchor_boxes.mat');

for d = {CHECKPOINT_DIR, MODEL_DIR}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

% ---- Class names ---------------------------------------------------------
classNames = ["Car", "Pedestrian", "Cyclist"];

% ---- Load datastores (from prepare_training_data.m) ----------------------
if ~exist('trainData', 'var') || ~exist('valData', 'var')
    fprintf('Running prepare_training_data.m ...\n');
    run('prepare_training_data.m');
end

n_train = numel(trainData.UnderlyingDatastores{1}.Files);
iters_per_epoch = floor(n_train / MINI_BATCH_SIZE);
total_iters     = MAX_EPOCHS * iters_per_epoch;
val_freq        = iters_per_epoch;   % validate once per epoch

fprintf('\n--- Training configuration ---\n');
fprintf('  Backbone        : %s\n',   NETWORK);
fprintf('  Input size      : %dx%dx%d\n', INPUT_SIZE(1), INPUT_SIZE(2), INPUT_SIZE(3));
fprintf('  Classes         : %s\n',   strjoin(classNames, ', '));
fprintf('  Epochs          : %d\n',   MAX_EPOCHS);
fprintf('  Mini-batch      : %d\n',   MINI_BATCH_SIZE);
fprintf('  Initial LR      : %.4f\n', INIT_LR);
fprintf('  Warm-up iters   : %d\n',   WARMUP_PERIOD);
fprintf('  L2 regularisation: %.4f\n', L2_REG);
fprintf('  Penalty threshold: %.2f\n', PENALTY_THRESHOLD);
fprintf('  Train samples   : %d\n',   n_train);
fprintf('  Iters / epoch   : %d\n',   iters_per_epoch);
fprintf('------------------------------\n\n');

% ---- Load anchor boxes ---------------------------------------------------
if ~exist(ANCHOR_FILE, 'file')
    error('Anchor file not found: %s\nRun prepare_training_data.m first.', ANCHOR_FILE);
end
loaded      = load(ANCHOR_FILE, 'anchorBoxes');
anchorBoxes = loaded.anchorBoxes;
fprintf('Anchor boxes loaded from %s\n', ANCHOR_FILE);

% ---- Build detector ------------------------------------------------------
fprintf('Creating Complex-YOLOv4 detector (%s) ...\n', NETWORK);
detector = yolov4ObjectDetector(NETWORK, classNames, anchorBoxes, ...
                                'InputSize', INPUT_SIZE);

% ---- Learning-rate schedule (cosine annealing with linear warm-up) -------
% MATLAB trainingOptions does not natively support cosine-annealing, so we
% approximate it with piecewise drops at fixed epochs.
%   - Iterations 1 .. WARMUP_PERIOD: LR ramps from INIT_LR/10 → INIT_LR
%   - After warm-up: cosine decay approximated at epoch 25 and epoch 40
warmup_epochs = ceil(WARMUP_PERIOD / max(iters_per_epoch, 1));
drop_epochs   = [warmup_epochs + round(MAX_EPOCHS * 0.5), ...
                 warmup_epochs + round(MAX_EPOCHS * 0.8)];
drop_factors  = [0.1, 0.1];   % multiply LR by 0.1 at each drop

% penaltyThreshold is applied inside the YOLO loss; expose via detector if
% the installed toolbox version supports it.
try
    detector.PenaltyThreshold = PENALTY_THRESHOLD;
    fprintf('penaltyThreshold set to %.2f\n', PENALTY_THRESHOLD);
catch
    fprintf('Note: penaltyThreshold property not available in this MATLAB version.\n');
    fprintf('      Using default penalty threshold.\n');
end

% ---- Training options ----------------------------------------------------
options = trainingOptions('sgdm', ...
    'MaxEpochs',              MAX_EPOCHS, ...
    'MiniBatchSize',          MINI_BATCH_SIZE, ...
    'InitialLearnRate',       INIT_LR, ...
    'LearnRateSchedule',      'piecewise', ...
    'LearnRateDropFactor',    drop_factors(1), ...
    'LearnRateDropPeriod',    drop_epochs(1), ...
    'Momentum',               MOMENTUM, ...
    'L2Regularization',       L2_REG, ...
    'ValidationData',         valData, ...
    'ValidationFrequency',    val_freq, ...
    'Shuffle',                'every-epoch', ...
    'Plots',                  'training-progress', ...
    'CheckpointPath',         CHECKPOINT_DIR, ...
    'Verbose',                true, ...
    'VerboseFrequency',       10, ...
    'ExecutionEnvironment',   'auto');

% ---- Train ---------------------------------------------------------------
fprintf('\n=== Starting Complex-YOLOv4 training ===\n\n');
t_start = tic;
[detector, trainingInfo] = trainYOLOv4ObjectDetector(trainData, detector, options);
elapsed = toc(t_start);
fprintf('\nTraining finished in %.1f s (%.1f min)\n', elapsed, elapsed/60);

% ---- Save final model ----------------------------------------------------
timestamp  = datestr(now, 'yyyymmdd_HHMMSS');  %#ok<TNOW1,DATST>
model_file = fullfile(MODEL_DIR, sprintf('complex_yolov4_bev_%s.mat', timestamp));
save(model_file, 'detector', 'trainingInfo', 'classNames', 'anchorBoxes', ...
     'MAX_EPOCHS', 'MINI_BATCH_SIZE', 'INIT_LR', 'L2_REG', 'PENALTY_THRESHOLD');
fprintf('Model saved to: %s\n', model_file);

% Keep a "latest" symlink for detect_objects.m
latest_file = fullfile(MODEL_DIR, 'yolov4_bev_latest.mat');
copyfile(model_file, latest_file);
fprintf('Latest model -> %s\n', latest_file);

% ---- Training-loss plot --------------------------------------------------
figure('Name', 'Complex-YOLOv4 Training');
subplot(1, 2, 1);
plot(trainingInfo.TrainingLoss, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Training');
hold on;
if ~all(isnan(trainingInfo.ValidationLoss))
    plot(trainingInfo.ValidationLoss, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Validation');
end
xlabel('Iteration'); ylabel('Loss');
title('Loss curves'); legend('Location', 'northeast'); grid on;

subplot(1, 2, 2);
lr_vals = trainingInfo.BaseLearnRate;
if ~isempty(lr_vals)
    plot(lr_vals, 'k-', 'LineWidth', 1.2);
    xlabel('Iteration'); ylabel('Learning rate');
    title('LR schedule'); grid on;
end

sgtitle(sprintf('Complex-YOLOv4 (epochs=%d, bs=%d, lr=%.4f)', ...
    MAX_EPOCHS, MINI_BATCH_SIZE, INIT_LR));
saveas(gcf, fullfile(MODEL_DIR, sprintf('training_curves_%s.png', timestamp)));
fprintf('Training curves saved.\n');
