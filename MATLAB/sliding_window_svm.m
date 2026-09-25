clear;
close all;

%% Image Paths
test_image_paths = {'im1.jpg', 'im2.jpg', 'im3.jpg', 'im4.jpg'};

%% Parameters
sample_rate = 1;          % Sampling rate for image loading
kFolds = 6;               % Number of cross-validation folds
target_size = [28, 21];   % Sliding window size
step_size = 4;            % Step size for the sliding window
scales = [1.1, 1, 0.9, 0.8] ;  % Scales for multi-scale detection
IoU_threshold = 0.2;      % Intersection over Union threshold for NMS

%% Load and Preprocess Training Data
[images, labels, original_dimensions] = loadFaceImages('face_train.cdataset', sample_rate);
disp('Training data loaded.');

% Resize training patches to 32x32
numImages = size(images, 1);
featureLength = length(hog_feature_vector(imresize(reshape(images(1, :), original_dimensions(1, 1), []), target_size)));
vectors = zeros(numImages, featureLength);

for i = 1:numImages
    image = reshape(images(i, :), original_dimensions(i, 1), []); 
    resized_image = imresize(image, target_size);
    vectors(i, :) = hog_feature_vector(resized_image);  % Extract features
end
disp('Feature extraction for training data completed.');

% Convert labels from 0 to -1 if necessary
if any(labels == 0)
    labels(labels == 0) = -1;
end

%% Cross-Validation Training and Model Evaluation
cv = cvpartition(labels, 'KFold', kFolds);
allPredictions = zeros(size(labels));
allScores = zeros(size(labels));
allActual = labels;

for fold = 1:kFolds
    fprintf('Processing Fold %d/%d...\n', fold, kFolds);

    trainIdx = training(cv, fold);
    testIdx = test(cv, fold);

    X_train = vectors(trainIdx, :);
    y_train = labels(trainIdx);
    X_test = vectors(testIdx, :);
    y_test = labels(testIdx);

    % Normalize training data
    [X_train_norm, mu, sigma] = zscore(X_train);

    % Normalize test data using training parameters
    X_test_norm = (X_test - mu) ./ sigma;

    % Train the SVM
    kernel = "gaussian";
    kerneloption = 10;
    modelSVM = SVMtraining(X_train_norm, y_train, kernel, kerneloption);
    modelSVM.mu = mu;
    modelSVM.sigma = sigma;
end

for i = 1:length(test_image_paths)
    test_image_path = test_image_paths{i};
    %% Sliding Window Detection with Multi-Scale
    test_image = imread(test_image_path);
    if size(test_image, 3) == 3  % Check if the image is RGB
        gray_image = rgb2gray(test_image);
    else
        gray_image = test_image;  % Already grayscale
    end
    
    fig = figure;
    imshow(test_image);
    hold on;
    tic;
    face_count = 0;
    detected_boxes = [];  % Array to store bounding boxes
    scores = [];          % Array to store corresponding scores
    
    for scale = scales
        scaled_image = imresize(gray_image, scale);
        [scaled_height, scaled_width] = size(scaled_image);
    
        for y = 1:step_size:(scaled_height - target_size(1) + 1)
            for x = 1:step_size:(scaled_width - target_size(2) + 1)
                % Extract current window
                current_window = scaled_image(y:y+target_size(1)-1, x:x+target_size(2)-1);
    
                % Extract features and normalize
                feature_vector = hog_feature_vector(current_window);
                feature_vector_norm = (feature_vector - modelSVM.mu) ./ modelSVM.sigma;
    
                % Classify using SVM
                [is_face, pred_val] = SVMTesting(feature_vector_norm, modelSVM);
    
                % If it's a face, save the bounding box and its score
                if is_face == 1
                    % Map back to original scale
                    orig_x = x / scale;
                    orig_y = y / scale;
                    orig_width = target_size(2) / scale;
                    orig_height = target_size(1) / scale;
    
                    % Store detected box and score
                    detected_boxes = [detected_boxes; orig_x, orig_y, orig_width, orig_height];
                    scores = [scores; pred_val];
                end
            end
        end
    end
    
    fprintf('SVM processing complete');
    
    %% Non-Maximum Suppression (NMS) to remove duplicate boxes
    if ~isempty(detected_boxes)
        % 'detected_boxes' is an Nx4 matrix with bounding boxes [x, y, w, h]
        % 'scores' is a column vector with the confidence scores for each detected box
        
        % Perform Non-Maximum Suppression using selectStrongestBbox
        % 'iouThreshold' is the IoU threshold for NMS (e.g., 0.3)
        
        % The function returns 'keep' as indices of boxes to keep after NMS
        [selected_boxes, selected_scores] = selectStrongestBbox(detected_boxes, scores, 'OverlapThreshold', IoU_threshold);
        
        % Update 'detected_boxes' and 'scores' with the selected boxes and scores
        detected_boxes = selected_boxes;
        scores = selected_scores;
    end
    
    
    
    fprintf('NMS processing complete');
    elapsed_time =toc;
    %% Draw the final boxes
    for i = 1:size(detected_boxes, 1)
        box = detected_boxes(i, :);
        rectangle('Position', box, 'EdgeColor', 'g', 'LineWidth', 0.5);
        
        % Add label to the detected face
        label = sprintf('%.2f', scores(i));  % Format the score as needed (e.g., %.2f for 2 decimal places)
        text(box(1) + 8, box(2) + 8, label, 'Color', 'yellow', 'FontSize', 4, 'FontWeight', 'bold');
    end
    
fprintf('Number of faces detected: %d | Processing Time: %.2f seconds\n', size(detected_boxes, 1), elapsed_time);
    
    exportgraphics(fig, ['output_' test_image_path], 'Resolution', 800);

end

