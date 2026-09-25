numTrees = 10;
maxSplit = 2;
train_file = 'face_train.cdataset';
test_file = 'face_test.cdataset';

[val_acc, train_time] = trainingRandomForestHaar(numTrees, maxSplit, train_file); 
[time,test_acc ] = testingRandomForestWithTiming2(test_file);

fprintf('Validation Accuracy: %.2f%%\n', val_acc * 100);
fprintf('Test Accuracy: %.2f%%\n', test_acc * 100);
fprintf('Time Taken: %.2f seconds\n', time);
fprintf('Training Time Taken: %.2f seconds\n', train_time);