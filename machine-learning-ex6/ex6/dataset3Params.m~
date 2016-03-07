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

C1 = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma1 = [0.01;0.03;0.1;0.3;1;3;10;30];

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

err = 999999;

m = size(C1,1);


for i = 1:m
	for j = 1:m
		model= svmTrain(X, y, C1(i), @(x1, x2) gaussianKernel(x1, x2, sigma1(j)));
		predictions = svmPredict(model, Xval);
		temp_err = mean(double(predictions ~= yval));
		
		%fprintf(" error for this set of C = %0.5f  and sigma = %0.5f is %0.5f \n",C1(i),sigma1(j),err);
		if (temp_err < err)
			err = temp_err;
			C = C1(i);
			sigma = sigma1(j);
		endif	
		
	endfor;
endfor;		    






% =========================================================================

end
