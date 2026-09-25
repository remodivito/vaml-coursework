function [elapsed_time, accuracy] = testingRandomForestWithTiming2(test_file)
% Load images of faces from provided test dataset
% Also loads ground truth labels
sample_rate = 1;
[images, labels, original_dimensions] = loadFaceImages(test_file, sample_rate);

% Perform feature extraction, converting images into feature vectors
vectors = [];
for i = 1:size(images, 1)
    image = reshape(images(i, :), original_dimensions(i, 1), []);
    vector = gabor_feature_vector(image);
    vectors = [vectors; vector];
end

% Load the trained Random Forest model
load('trained_forest_model.mat', 'forest_model');

% Measure the prediction timing
num_samples = size(vectors, 1);
predictions = zeros(num_samples, 1);
start_time = tic;

for i = 1:num_samples
    predictions(i) = str2double(predict(forest_model, vectors(i, :))); % Convert cell array to numeric
end

elapsed_time = toc(start_time);

% Calculate accuracy
accuracy = sum(predictions == labels) / num_samples;

fprintf('Testing Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Time taken for predictions: %.2f seconds\n', elapsed_time);

end