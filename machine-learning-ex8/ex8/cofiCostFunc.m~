function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
s = 0;
reg1 = 0;
reg2 = 0; 

X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);
grad1 = zeros(num_users,num_features);
grad2 = zeros(num_movies,num_features);


for i = 1:num_movies
	for j = 1:num_users
		if(R(i,j) == 1)		
		 s = sum(Theta(j,:) * X(i,:)');
		 J = J + (Y(i,j) - s).^2;
		 s3 = 0;
		 for k = 1:num_features
		  grad1(j,k) = grad1(j,k) + (s - Y(i,j)) * X(i,k);
		  grad2(i,k) = grad2(i,k) + (s - Y(i,j)) * Theta(j,k);
		 endfor;		 
		endif;
	endfor;
endfor;

for j = 1:num_users
	for k = 1:num_features
		reg1 = reg1 + Theta(j,k)^2; 
	endfor;
endfor;

for i = 1:num_movies
	for k = 1:num_features
		reg2 = reg2 + X(i,k)^2; 
	endfor;
endfor;

reg1 = (reg1 * lambda)/2;
reg2 = (reg2 * lambda)/2;

J = J / 2;


J = J + reg1 + reg2;

Theta_grad = grad1 + lambda * Theta;
X_grad = grad2 + lambda * X;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
