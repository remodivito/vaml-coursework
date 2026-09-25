function testingRandomForest2(imageFolderPath)
    % Load the trained random forest model
    load('randomForestModel500.mat', 'randomForestModel');  % Loads the random forest structure with trees

    % Get a list of all images in the specified folder
    imageFiles = dir(fullfile(imageFolderPath, '*.jpg'));  % Adjust extension if needed
    if isempty(imageFiles)
        disp('No images found in the specified folder.');
        return;
    end
    
    % Loop through each image in the folder
    for i = 1:length(imageFiles)
        % Construct the full path to the image
        imagePath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        
        % Load and preprocess the image
        img = imread(imagePath);
        
        % Convert to grayscale if the image is RGB
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
        
        % Display the result for the current image
        if isFace == 1
            fprintf('Image %s: The image contains a face.\n', imageFiles(i).name);
        else
            fprintf('Image %s: The image does not contain a face.\n', imageFiles(i).name);
        end
    end
end
