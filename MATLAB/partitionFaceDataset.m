function [trainImages, trainLabels, testImages, testLabels] = partitionFaceDataset(filePath, trainRatio, sampling)
    % Load the dataset
    [images, labels, ~] = loadFaceImages(filePath, sampling);

    % Ensure reproducibility
    rng('default');

    % Shuffle the dataset
    numSamples = size(images, 1);
    indices = randperm(numSamples);
    images = images(indices, :);
    labels = labels(indices, :);

    % Determine the split point
    numTrain = round(numSamples * trainRatio);

    % Partition the data
    trainImages = images(1:numTrain, :);
    trainLabels = labels(1:numTrain, :);
    testImages = images(numTrain+1:end, :);
    testLabels = labels(numTrain+1:end, :);
end
