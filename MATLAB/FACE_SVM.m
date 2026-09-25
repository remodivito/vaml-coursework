clear
close all

%% Parameters
sample_rate = 4;          % Sampling rate for image loading
kFolds = 5;               % Number of cross-validation folds (e.g., 5-fold)

%% Load All Data

% Load all images and labels from the dataset
[images, labels, original_dimensions] = loadFaceImages('all_faces.cdataset', sample_rate);

disp('All data loaded.');

%% Feature Extraction

% Preallocate the vectors matrix for efficiency
numImages = size(images, 1);
% Assuming hog_feature_vector returns a fixed-length vector
% Compute the feature length using the first image
firstImage = reshape(images(1, :), original_dimensions(1, 1), []);
firstVector = hog_feature_vector(firstImage);
featureLength = length(firstVector);

vectors = zeros(numImages, featureLength);

for i = 1:numImages
    image = reshape(images(i, :), original_dimensions(i, 1), []);
    vectors(i, :) = hog_feature_vector(image);
end
disp('Feature extraction completed.');

%% Cross-Validation Setup

% Convert labels to binary (-1 and 1) if they are not already
% Adjust this section if your labels are different
uniqueLabels = unique(labels);
if length(uniqueLabels) ~= 2
    error('ROC curve requires binary classification. Please ensure there are exactly two classes.');
end

% Identify the positive class (assume the higher label is positive)
positiveClass = max(uniqueLabels);

% Convert labels to binary (-1 and 1) if necessary
if any(labels == 0)
    labels(labels == 0) = -1;
end

% Generate cross-validation indices
cvIndices = crossvalind('Kfold', labels, kFolds);

% Initialize variables to store results
allPredictions = zeros(size(labels));
allActual = labels;
allScores = zeros(size(labels));  % To store decision scores

%% Cross-Validation Loop

for fold = 1:kFolds
    fprintf('Processing Fold %d/%d...\n', fold, kFolds);
    
    % Split data into training and validation sets
    testIdx = (cvIndices == fold);
    trainIdx = ~testIdx;
    
    X_train = vectors(trainIdx, :);
    y_train = labels(trainIdx);
    
    X_val = vectors(testIdx, :);
    y_val = labels(testIdx);
    
    %% Normalize Training Data
    [X_train_norm, mu, sigma] = zscore(X_train);
    
    %% Normalize Validation Data using Training Parameters
    X_val_norm = (X_val - mu) ./ sigma;
    
    %% Train the SVM Model
    kernel = "gaussian";       % Specify kernel type
    kerneloption = 10;         % Specify kernel option (e.g., bandwidth for Gaussian)
    modelSVM = SVMtraining(X_train_norm, y_train, kernel, kerneloption);
    
    % Store normalization parameters in the model (optional for consistency)
    modelSVM.mu = mu;
    modelSVM.sigma = sigma;
    
    %% Validate the Model
    predictions = zeros(size(X_val_norm, 1), 1);
    scores = zeros(size(X_val_norm, 1), 1);  % To store decision scores
    
    for i = 1:size(X_val_norm, 1)
        test_vector = X_val_norm(i, :);
        [predictions(i), scores(i)] = SVMTesting(test_vector, modelSVM);
        % Assuming SVMTesting is modified to return [prediction, score]
    end
    
    % Store predictions and scores
    allPredictions(testIdx) = predictions;
    allScores(testIdx) = scores;
end

%% Evaluation

% Calculate accuracy
comparison = (allActual == allPredictions);
Accuracy = sum(comparison) / length(comparison);

fprintf('Cross-Validation Accuracy: %.2f%%\n', Accuracy * 100);

%% ROC Curve

% Ensure that scores are aligned with the positive class
% Depending on your SVM implementation, higher scores should correspond to the positive class
% Adjust if necessary

% Convert labels to binary format for perfcurve
binaryLabels = (allActual == positiveClass);

% Compute ROC curve and AUC
[X_roc, Y_roc, T, AUC] = perfcurve(binaryLabels, allScores, true);

% Plot ROC curve
figure;
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
hold on;
plot([0,1], [0,1], 'k--');  % Diagonal line
hold off;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.2f)', AUC));
grid on;

% Confusion Matrix
confMat = confusionmat(allActual, allPredictions);
disp('Confusion Matrix:');
disp(confMat);

% Precision, Recall, F1-Score (if binary classification)
TP = confMat(positiveClass, positiveClass);
TN = confMat(-positiveClass, -positiveClass);
FP = confMat(-positiveClass, positiveClass);
FN = confMat(positiveClass, -positiveClass);

Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
F1 = 2 * (Precision * Recall) / (Precision + Recall);

fprintf('Precision: %.2f%%\n', Precision * 100);
fprintf('Recall: %.2f%%\n', Recall * 100);
fprintf('F1-Score: %.2f%%\n', F1 * 100);
fprintf('AUC: %.2f\n', AUC);
