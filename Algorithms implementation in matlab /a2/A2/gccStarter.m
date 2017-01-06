function [estimatedTargets, error] = gccStarter(c1_train, c2_train, testInputs, testTargets)
% Use the function learnGCCmodel to learn a Gaussian Class-Conditional
% model for classification.
% Use the function gccClassify for evaluating the decision function
% to perform classification.
% Then test the model under various conditions.

[p1, m1, m2, C1, C2]=learnGCCmodel(c1_train', c2_train');
total = size(testTargets, 2);
estimatedTargets = zeros(total, 2);
count = 0;
error = 0;
figure();
for i=1:size(testInputs,2)
    count = count + 1;
    class = gccClassify(testInputs(:, i), p1, m1, m2, C1, C2);
    estimatedTargets(count, :) = class;
    if class ~= transpose(testTargets(:,i))
        error = error + 1;
        plot(testInputs(1,i), testInputs(2,i),'rs');hold on; 
    else
        if class(1,1) == 1
            plot1=plot(testInputs(1,i), testInputs(2,i),'bx');hold on;
        elseif class(1,1) == 0
            plot2=plot(testInputs(1,i), testInputs(2,i),'go');hold on;
        end
    end
end

xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('GCC fitting model for generic data')
hold on;
disp(total);
error = error/total;
disp(error)
end