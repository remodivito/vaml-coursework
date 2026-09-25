function [accuracy] = trainingRandomForest2(num_trees, max_splits, folder_path)
% Modified version of trainingRandomForest.m incorporating loadFaceImages functionality

% Load images of faces from provided training dataset
% Also loads ground truth labels
sample_rate = 1;
[images, labels, original_dimensions] = loadFaceImages('face_train.cdataset', sample_rate);

% Perform feature extraction, converting images into feature vectors
vectors = [];
for i = 1:size(images, 1)
    image = reshape(images(i, :), original_dimensions(i, 1), []);
    vector = gabor_feature_vector(image);
    vectors = [vectors; vector];
end

% Split data into training and validation sets
cv = cvpartition(labels, 'HoldOut', 0.2);
X_train = vectors(training(cv), :);
y_train = labels(training(cv));
X_val = vectors(test(cv), :);
y_val = labels(test(cv));

% Train Random Forest Classifier
forest_model = TreeBagger(num_trees, X_train, y_train, 'OOBPrediction', 'On', 'Method', 'classification', 'MaxNumSplits', max_splits);

% Store the trained Random Forest model
save('trained_forest_model.mat', 'forest_model');

% Validate the model
predictions = str2double(predict(forest_model, X_val)); % Convert cell array to numeric
accuracy = sum(predictions == y_val) / length(y_val);

fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
end
