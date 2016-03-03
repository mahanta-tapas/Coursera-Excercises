function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];

 
z2 = X * Theta1';
%z1 is 3x2
hidden_layer = sigmoid(z2);

hidden_layer = [ones(m, 1) hidden_layer];
%size(hidden_layer)
%size(Theta2)
z3 = hidden_layer * Theta2';

hypt = sigmoid(z3); 





%p(501,:)
%[x , ix] = max(p');

%p = ix';

%hypt = zeros(m,num_labels);

%for i = 1:m
%r = p(i);
%hypt(i,r) = hypt(i,r) + 1;
%endfor;

%hypt(501,:)
%size(hypt)
% hypt size is 5000x1

output = zeros(m,num_labels);

for i = 1:m

k = y(i); 
output(i,k) = output(i,k) + 1; 

endfor;


cost = 0;

for i = 1:m
	for j= 1:num_labels
	cost = cost + (output(i,j)*log(hypt(i,j)) + (1-output(i,j))*log(1-hypt(i,j)));
	%fprintf("cost is %f\n",cost);
	endfor;
endfor;

J = -cost/m;

fprintf('now j is %f \n' ,J);


%theta1 25 x 401
%theta2 10 x 26

sum1 = 0.0;
sum2 = 0.0;
sum3 = 0.0;
%size(Theta1)
%size(Theta2)
%Theta2
%input_layer_size
%hidden_layer_size

for i = 1:hidden_layer_size
	for j = 2:input_layer_size+1
	    sum1 = sum1 + Theta1(i,j)^2;
	endfor;
endfor;		
		
for i = 1:num_labels
	for j = 2:hidden_layer_size+1
	    sum2 = sum2 + Theta2(i,j)^2;
	endfor;
endfor;		
		
sum3 = (( sum1 + sum2 ) * lambda)/(2 * m);

J = J + sum3;

fprintf('After j is %f \n' ,J);
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d3 = zeros(num_labels,m);
d2 = zeros(m,hidden_layer_size);


delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

a1 = X;
a2 = hidden_layer;
a3 = hypt;




d3 =  a3 - output

a2
size(Theta2(:,2:end))
size(d3)
size(z2)

d2 = (Theta2(:,2:end)' * d3') .* sigmoidGradient(z2)';	
d2 = d2';

delta2 = delta2 + d3' * a2; 

delta1 = delta1 + d2' * a1;
       

       


Theta1_grad = delta1/m;
Theta2_grad = delta2/m;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end) .*(lambda/m);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end) .*(lambda/m);






% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
