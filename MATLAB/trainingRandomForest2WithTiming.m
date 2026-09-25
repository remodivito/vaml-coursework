function [val_accuracy, train_time] = trainingRandomForest2(num_trees, max_splits, train_file)
% Train a Random Forest model using the specified parameters and measure training time

% Load training images from provided dataset
sample_rate = 1;
[train_images, train_labels, train_original_dimensions] = loadFaceImages(train_file, sample_rate);

% Perform feature extraction on training data
train_vectors = [];
for i = 1:size(train_images, 1)
    image = reshape(train_images(i, :), train_original_dimensions(i, 1), []);
    vector = gabor_feature_vector(image);
    train_vectors = [train_vectors; vector];
end

% Split data into training and validation sets
cv = cvpartition(size(train_labels, 1), 'HoldOut', 0.2);
X_train = train_vectors(training(cv), :);
y_train = train_labels(training(cv));
X_val = train_vectors(test(cv), :);
y_val = train_labels(test(cv));

% Measure training time
train_start_time = tic;

% Train Random Forest Classifier
forest_model = TreeBagger(num_trees, X_train, y_train, 'OOBPrediction', 'On', 'Method', 'classification', 'MaxNumSplits', max_splits);

% Record training time
train_time = toc(train_start_time);

% Validate the model on validation set
val_predictions = str2double(predict(forest_model, X_val)); % Convert cell array to numeric
val_accuracy = sum(val_predictions == y_val) / length(y_val);

fprintf('Validation Accuracy: %.2f%%', val_accuracy * 100);
fprintf('Training Time: %.2f seconds\', train_time);

% Save the trained model
save('trained_forest_model.mat', 'forest_model');
end
