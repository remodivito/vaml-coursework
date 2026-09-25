function testingRandomForestWithTiming(testImageFolderPath)
    % Load the trained random forest model
    load('randomForestModel500.mat', 'randomForestModel');  % Load the trained model

    % Define the range of numTrees values to test
    numTreesValues = [7, 9, 10, 400, 500];
    
    % Get a list of all .jpg images in the specified folder
    testImages = dir(fullfile(testImageFolderPath, '*.jpg'));
    if isempty(testImages)
        disp('No images found in the specified folder.');
        return;
    end
    
    % Loop through each specified number of trees
    for nt = numTreesValues
        % Check if the current number of trees exceeds the trained model's size
        if nt > randomForestModel.NumTrees
            fprintf('Skipping numTrees = %d (exceeds the trained model''s number of trees)\n', nt);
            continue;
        end
        
        fprintf('\nTesting with numTrees = %d\n', nt);
        
        % Initialize timing variables
        totalTime = 0;
        numImages = length(testImages);
        
        % Loop through each test image
        for i = 1:numImages
            % Construct the full path to the image
            imagePath = fullfile(testImages(i).folder, testImages(i).name);
            
            % Load and preprocess the image
            img = imread(imagePath);
            if size(img, 3) == 3
                img = rgb2gray(img);  % Convert to grayscale if the image is RGB
            end
            
            % Extract features using the Gabor filter function
            featureVector = gabor_feature_vector(img);

            % Start timing for the prediction
            tic;
            
            % Initialize an array to collect votes from each tree
            votes = zeros(nt, 1);

            % Use only the specified number of trees to make a prediction
            for t = 1:nt
                tree = randomForestModel.Trees{t};
                votes(t) = predict(tree, featureVector);
            end
            
            % Aggregate votes by taking the majority vote
            isFace = mode(votes);  % Majority voting: 1 means face, 0 means non-face
            
            % Stop timing for this prediction and add to total time
            predictionTime = toc;
            totalTime = totalTime + predictionTime;
            
            % Display the result for the current image
            if isFace == 1
                fprintf('Image %s: The image contains a face. Prediction time: %.4f seconds\n', testImages(i).name, predictionTime);
            else
                fprintf('Image %s: The image does not contain a face. Prediction time: %.4f seconds\n', testImages(i).name, predictionTime);
            end
        end
        
        % Calculate and display the average prediction time
        avgTime = totalTime / numImages;
        fprintf('Total testing time for numTrees = %d: %.4f seconds\n', nt, totalTime);
        fprintf('Average prediction time per image for numTrees = %d: %.4f seconds\n', nt, avgTime);
    end
end
