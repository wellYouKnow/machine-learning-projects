function [p1, m1, m2, C1, C2] = learnGCCmodel(x1, x2)
% 
% Inputs
%   x1 - training exemplars from class 1, one exemplar per row
%   x2 - training exemplars from class 2, one exemplar per row
%
% Outputs
%   p1 - prior probability for class 1
%   m1 - mean of Gaussian measurement likelihood for class 1
%   m2 - mean of Gaussian measurement likelihood for class 2
%   C1 - covariance of Gaussian measurement likelihood for class 1
%   C2 - covariance of Gaussian measurement likelihood for class 2
%

% only for test
%x1 = rand(3);
%x2 = rand(5,3);

n1 = size(x1, 1);
n2 = size(x2, 1);
p1 = n1/(n1+n2);

m1 = sum(x1,1)'/n1;
m2 = sum(x2,1)'/n2;

% maximum likelihood estimator: divides sum by n instead of n-1
C1 = cov(x1, 1);
C2 = cov(x2, 1);
end
