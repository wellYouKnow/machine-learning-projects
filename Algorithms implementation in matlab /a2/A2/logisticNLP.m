function [ll, dll_dw, dll_db] = logisticNLP(x1, x2, w, b, alpha)
% [ll, dll_dw, dll_db] = logisticNLP(x1, x2, w, b, alpha)
% 
% Inputs:
%   x1 - array of exemplar measurement vectors for class 1.
%   x2 - array of exemplar measurement vectors for class 2.
%   w - an array of weights for the logistic regression model.
%   b - the bias parameter for the logistic regression model.
%   alpha - weight decay parameter
% Outputs:
%   ll - negative log probability (likelihood) for the data 
%        conditioned on the model (ie w).
%   dll_dw - gradient of negative log data likelihood wrt b
%   dll_db - gradient of negative log data likelihood wrt b


% YOUR CODE GOES HERE.

% posterior class probability
p1 = logistic(x1, w, b);
p2 = logistic(x2, w, b);

n1 = size(x1, 2);
n2 = size(x2, 2);

x = [x1 x2];
% label for x1 is 1, label for x2 is 0 in LR algorithm
y = [ones(1, n1) zeros(1, n2)];

ll = 0.5*alpha*transpose(w)*w-(sum(log(p1))+sum(log(1-p2)));

dll_dw = w/alpha - x*(y - [p1 p2])';
dll_db = -sum(y - [p1 p2]);
end
