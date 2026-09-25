function randomForestModel = trainingRandomForest(numTrees)
    % Load Gabor filter bank (assuming gabor.mat has required filters)
    load('gabor.mat');  % Load the Gabor filters
    
    % Initialize arrays to hold features and labels
    features = [];
    labels = [];
    
    % Load images and labels
    faceImages = imageDatastore("C:\Users\benir\Documents\MATLAB\images\face", 'FileExtensions', '.png');
    notFaceImages = imageDatastore("C:\Users\benir\Documents\MATLAB\images\non-face", 'FileExtensions', '.png');
    
    % Process face images
    for i = 1:numel(faceImages.Files)
        img = readimage(faceImages, i);
        featureVector = gabor_feature_vector(img);  % Extract Gabor features
        features = [features; featureVector];
        labels = [labels; 1];  % Label for face = 1
    end
    
    % Process non-face images
    for i = 1:numel(notFaceImages.Files)
        img = readimage(notFaceImages, i);
        featureVector = gabor_feature_vector(img);  % Extract Gabor features
        features = [features; featureVector];
        labels = [labels; 0];  % Label for not-face = 0
    end
    
    % Initialize Random Forest parameters
    trees = cell(numTrees, 1);  % Store each trained tree

    % Train the random forest with bootstrap sampling
    for t = 1:numTrees
        % Bootstrap sample indices
        sampleIdx = randsample(1:size(features, 1), size(features, 1), true);
        
        % Create training set from bootstrap sample
        bootFeatures = features(sampleIdx, :);
        bootLabels = labels(sampleIdx);
        
        % Train decision tree on the bootstrap sample 
        tree = fitctree(bootFeatures, bootLabels, 'MaxNumSplits', 10, 'SplitCriterion', 'gdi');  % Limiting splits to prevent overfitting and specifying Gini index
        trees{t} = tree;
    end

    % Store the random forest model
    randomForestModel = struct('Trees', {trees}, 'NumTrees', numTrees);
    
    % Save the trained model using version 7.3 to support the large file
    save('randomForestModel500.mat', 'randomForestModel', '-v7.3');
end
