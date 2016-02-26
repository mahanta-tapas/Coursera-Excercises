function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

n = size(theta,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i = 1:m

	s = 0;
	for j = 1:n

	s = s + X(i,j) * theta(j);
	endfor

	

	h = sigmoid(s);
	%fprintf('h is %f \n',h);
	v = (y(i) * log(h) + (1 - y(i)) * log(1-h));
	%fprintf('v is %f \n',v);
	J = J + v;
	
	for j1 = 1:n
	grad(j1) = grad(j1) + (h - y(i)) * X(i,j1);
	endfor
 
	
endfor
	reg =  0;
	for k = 2:n
	reg = reg + theta(k) * theta(k);								
	endfor
	fprintf('reg is %f \n',reg);
	reg = (reg * lambda) / 2;
	J = (-J  + reg ) / m;

	grad(1) = grad(1)/m;

	for j1 = 2:n
	grad(j1) = (grad(j1) + lambda * theta(j1)) / m;
	endfor


end;


% =============================================================

