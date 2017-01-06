function result = logisticStarter(c1_train, c2_train, testInputs, testTargets)
% Use learnLogReg() to test performance on various datasets.

alphas = [0.25, 0.5, 1, 2, 5];

result = zeros(5, 2);
total = size(testTargets, 2);
count = 0;
% numel: number of elements
for i = 1:numel(alphas)
    error = 0;
    count = count + 1;
    % get estimated w and b
    [w, b]=learnLogReg(c1_train, c2_train, alphas(i));
    p = logistic(testInputs, w, b);
    for j= 1:total
        if round(p(j))~=testTargets(1, j)
            error = error +1;
        end
    end
    result(count, :) = [alphas(i) error/total];
end

end