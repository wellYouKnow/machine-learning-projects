function result = knnStarter(trainingInputs, trainingTargets, testInputs, testTargets)
%
% Inputs:
%   traingingInputs: array of training exemplars, one exemplar per row
%   traingingTargets: idenicator vector per row
%   testingInputs: array of testing exemplars, one exemplar per row
%   testingTargets: idenicator vector per row
%
% Outputs:
%   A 10*2 matrix where each row has the format: 
%   (the value of k, the test error for this k)  

% Use the function knnClassify to test performance on different datasets.
result = zeros(10, 2);
total = size(testTargets, 1);
count = 0;
% 0-1 loss function
for k=3:2:21
    error = 0;
    count = count + 1;
    for i = 1: total
        class = knnClassify(testInputs(i,:), k, trainingInputs, trainingTargets);
        if class ~= testTargets(i,:)
            error = error + 1;
        end
    end
    result(count, :) = [k error/total];
end
end