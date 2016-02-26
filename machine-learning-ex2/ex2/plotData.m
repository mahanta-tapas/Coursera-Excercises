function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure


% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

fprintf(["came here"]);

v = find(y == 1); 
n = find(y == 0);

plot(X(v, 1), X(v, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(n, 1), X(n, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);






% =========================================================================





end
