function isFace = testingRandomForest(imagePath)
    % Load the trained random forest model
    load('randomForestModel500.mat', 'randomForestModel');  % Loads the random forest structure with trees

    % Load the image and preprocess
    img = imread(imagePath);
    
    % Check if the image is grayscale or RGB, convert to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Extract features using the Gabor filter function
    featureVector = gabor_feature_vector(img);

    % Initialize an array to collect votes from each tree
    votes = zeros(randomForestModel.NumTrees, 1);

    % Use each tree to make a prediction
    for t = 1:randomForestModel.NumTrees
        tree = randomForestModel.Trees{t};
        votes(t) = predict(tree, featureVector);
    end
    
    % Aggregate votes by taking the majority vote
    isFace = mode(votes);  % Majority voting: 1 means face, 0 means not-face
    
    % Display result
    if isFace == 1
        fprintf('The image contains a face.\n');
    else
        fprintf('The image does not contain a face.\n');
    end
end
