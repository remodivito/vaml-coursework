function model = SVMtraining(images, labels, kernel, kerneloption)
% images: training data
% labels: training labels
% kernel: kernel type ('polyhomog', 'poly', 'gaussian', etc.)
% kerneloption: kernel parameter (e.g., degree for 'poly')
%addpath SVM-KM/

% Binary classification
model.type = 'binary';

% SVM software requires labels -1 or 1 for the binary problem
labels(labels == 0) = -1;

% Initialize and setup SVM parameters
lambda = 1e-4;  
C = 1;

% Calculate the support vectors
[xsup, w, w0, ~, ~, ~] = svmclass(images, labels, C, lambda, kernel, kerneloption, 1); 

% Create a structure encapsulating all the variables composing the model
model.xsup = xsup;
model.w = w;
model.w0 = w0;

model.param.kerneloption = kerneloption;
model.param.kernel = kernel;

end

