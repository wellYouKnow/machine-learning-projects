function class = knnClassify(test, k, trainingInputs, trainingTargets)
%
% Inputs:
%   test: test input vector
%   k: number of nearest neighbours to use in classification.
%   traingingInputs: array of training exemplars, one exemplar per row
%   traingingTargets: idenicator vector per row
%
% Basic Algorithm of kNN Classification
% 1) find distance from test input to each training exemplar,
% 2) sort distances
% 3) take smallest k distances, and use the median class among 
%    those exemplars to label the test input.

% only for test
%test = zeros(1,4);
%trainingInputs = ones(3,4);
%trainingTargets = zeros(3,2);

% make up a equal dimension test matrix to do matrix operation
m=size(trainingInputs, 1);
testMatrix = repmat(test, m, 1);
 
% find distance from test input
squaredDistance = sum((testMatrix-trainingInputs).^2, 2);

% sort distance and keep their corresponding label
combination = [squaredDistance trainingTargets];
[d1,d2] = sort(combination(:,1));
combination = combination(d2,:);

% take smallest k distances, and use the median class among 
% those exemplars to label the test input.
median = k/2;
% extreme case for k=1 
if k == 1
   class = combination(1, 2:3);
else
   class = combination(floor(median), 2:3);
end

end