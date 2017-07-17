function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
 % hypothesis h(theta) = g(x * theta); g is a sigmoid function

 h = sigmoid( X * theta);

 % j(theta) = (1/m) * ( sum(- y * log(h(theta)) - (1-y) * (log(1- h(theta))) ) 

 J = (1/m) * (-y' * log(h) - (1 - y)' * log(1-h));

 % grad is a vector of size theta with each term a partial derivate of J with theta_j
 % grad_j = (1/m) * (sum(h(theta)-y) * X_j)

 grad = (1/m) * X' * (h-y);


% =============================================================

end
