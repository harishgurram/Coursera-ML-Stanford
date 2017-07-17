function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values_C = [0.01 0.03 0.1 0.3 1 3 10 30]; % values to try for C
values_sigma = [0.01 0.03 0.1 0.3 1 3 10 30]; % values to try for sigma

% iterate over combinations of C and sigma to compute prediction error
% and find best combination of C and sigma to be used with SVM

minimum_error = Inf;

for i=1:length(values_C)
	for j = 1:length(values_sigma)
		Ci= values_C(i);
		sigmaj = values_sigma(j);
		% compute the SVM trained model with current values of C and sigma
		% find the predictions out of the model
		% compute the prediction error for the predictions
		model = svmTrain(X, y, Ci, @(x1,x2) gaussianKernel(x1,x2,sigmaj));
		predictions = svmPredict(model, Xval);
		prediction_error = mean(double(predictions ~= yval));

		if prediction_error < minimum_error
			minimum_error = prediction_error;
			C = Ci;
			sigma = sigmaj;
		end
	end
end


% =========================================================================

end
