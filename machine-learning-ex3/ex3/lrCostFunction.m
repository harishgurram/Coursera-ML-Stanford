function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% hypothesis h(theta) = g(x * theta); g is a sigmoid function

 h = sigmoid( X * theta);

 % j(theta) = (1/m) * ( sum(- y * log(h(theta)) - (1-y) * (log(1- h(theta))) )  + (Lambda/2* m) * sum(theta ^ 2)

 % as we should avoid regularizing theta_0 (theta_1 in octave), populate a new theta vector with theta_1 as 0

 theta_without_initialindex = theta(2:size(theta));

 theta_new = [0; theta_without_initialindex];

 J = (1/m) * (-y' * log(h) - (1 - y)' * log(1-h)) + (lambda /(2*m)) * theta_new' * theta_new;


 % grad is a vector of size theta with each term a partial derivate of J with theta_j
 % grad_j = (1/m) * (sum(h(theta)-y) * X_j) + (lambda/m) * theta;

 grad = (1/m) * X' * (h-y) + (lambda/m) * theta_new;


% =============================================================

grad = grad(:);

end
