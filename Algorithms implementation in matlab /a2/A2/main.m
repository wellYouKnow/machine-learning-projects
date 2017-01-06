%% Generic1/2: 
% This should be modified if the directory is different
load('generic1');

% c1_train and c2_train are one exp per column
c1_train_col = size(c1_train, 2);
c2_train_col = size(c2_train, 2);


trainingInputs = transpose([c1_train c2_train]);
trainingTargets = [ones(c1_train_col, 1) zeros(c1_train_col, 1);
    zeros(c2_train_col, 1) ones(c2_train_col, 1)];

% get size of data
c1_test_col = size(c1_test, 2);
c2_test_col = size(c2_test, 2); 

testInputs = transpose([c1_test c2_test]);
testTargets = [ones(c1_test_col, 1) zeros(c1_test_col, 1);
    zeros(c2_test_col, 1) ones(c2_test_col, 1)];

% apply Knn algorithm
g1Knn = knnStarter(trainingInputs, trainingTargets, testInputs, testTargets);

% apply logistic Regression
g1Log = logisticStarter(c1_train, c2_train, testInputs', testTargets');

% apply GCC model
[g1EstimatedTargetGCC, g1ErrorGcc] = gccStarter(c1_train, c2_train, testInputs', testTargets');
%%
%
%% Plot logistic and KNN model fit for generic1/2
%% GCC
[trainTarget, trainError] = gccStarter(c1_train, c2_train, trainingInputs', trainingTargets');
%% Logistic
% since some alphas has the same test error value, choose the 
% smallest alpha by occam's razor

% for testing data
alpha = 0.25;
total = size(testTargets, 1);
figure();
[w, b]=learnLogReg(c1_train, c2_train, alpha);
p = logistic(testInputs', w, b);
for j= 1:total
    if round(p(j))~=testTargets(j, 1)
        plot(testInputs(j, 1), testInputs(j, 2),'rs');hold on;
    else
        if round(p(j)) == 1
            plot1 = plot(testInputs(j, 1), testInputs(j, 2),'bx');hold on;
        elseif  round(p(j)) == 0
            plot2 = plot(testInputs(j, 1), testInputs(j, 2),'go');hold on;
        end
    end
end
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('Logistic fitting model for generic1 test data')
hold on;

%%
% for training data
alpha = 0.25;
total = size(trainingTargets, 1);
figure();
[w, b]=learnLogReg(c1_train, c2_train, alpha);
p = logistic(trainingInputs', w, b);
for j= 1:total
    if round(p(j))~=trainingTargets(j, 1)
        plot(trainingInputs(j, 1), trainingInputs(j, 2),'rs');hold on;
    else
        if round(p(j)) == 1
            plot1 = plot(trainingInputs(j, 1), trainingInputs(j, 2),'bx');hold on;
        elseif  round(p(j)) == 0
            plot2 = plot(trainingInputs(j, 1), trainingInputs(j, 2),'go');hold on;
        end
    end
end
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('Logistic fitting model for generic1 train data')
hold on;
%%
%KNN
% since some ks has the same test error value, choose the 
% smallest k by occam's razor

% for test data

k=3;
total = size(testTargets, 1);
figure();
 for i = 1: total
     class = knnClassify(testInputs(i,:), k, trainingInputs, trainingTargets);
     if class ~= testTargets(i,:)
        plot(testInputs(i, 1), testInputs(i, 2),'rs');hold on;
     else
        if class(1,1) == 1
            plot1 = plot(testInputs(i, 1), testInputs(i, 2),'bx');hold on;
        elseif class(1,1) == 0
            plot2 = plot(testInputs(i, 1), testInputs(i, 2),'go');hold on;
        end
    end
 end
 
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('KNN fitting model for generic1 test data')
hold on;

%% for training
k=3;
total =  size(trainingTargets, 1);
figure();
 for i = 1: total
     class = knnClassify(trainingInputs(i,:), k, trainingInputs, trainingTargets);
     if class ~= testTargets(i,:)
        plot(trainingInputs(i, 1), trainingInputs(i, 2),'rs');hold on;
     else
        if class(1,1) == 1
            plot1 = plot(trainingInputs(i, 1), trainingInputs(i, 2),'bx');hold on;
        elseif class(1,1) == 0
            plot2 = plot(trainingInputs(i, 1), trainingInputs(i, 2),'go');hold on;
        end
    end
 end
 
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('KNN fitting model for generic1 train data')
hold on;
%%
%
%% Basicly Generic2 is the same as generic1 but only different data value
load('generic2');

% c1_train and c2_train are one exp per column
c1_train_col = size(c1_train, 2);
c2_train_col = size(c2_train, 2);

trainingInputs = transpose([c1_train c2_train]);
trainingTargets = [ones(c1_train_col, 1) zeros(c1_train_col, 1);
    zeros(c2_train_col, 1) ones(c2_train_col, 1)];

c1_test_col = size(c1_test, 2);
c2_test_col = size(c2_test, 2);

testInputs = transpose([c1_test c2_test]);
testTargets = [ones(c1_test_col, 1) zeros(c1_test_col, 1);
    zeros(c2_test_col, 1) ones(c2_test_col, 1)];

% apply Knn algorithm to generic2
g2Knn = knnStarter(trainingInputs, trainingTargets, testInputs, testTargets);

% apply logistic Regression to generic2
g2Log = logisticStarter(c1_train, c2_train, testInputs', testTargets');

% apply GCC model 
[g2EstimatedTargetGcc, g2ErrorGcc] = gccStarter(c1_train, c2_train, testInputs', testTargets');
%%
%
%% Plot logistic, GCC and KNN model fit for generic1/2
%% GCC
[trainTarget, trainError] = gccStarter(c1_train, c2_train, trainingInputs', trainingTargets');

%%
% Logistic
alpha = 0.25;
total = size(testTargets, 1);
error = 0;
figure();
[w, b]=learnLogReg(c1_train, c2_train, alpha);
p = logistic(testInputs', w, b);
for j= 1:total
    if round(p(j))~=testTargets(j, 1)
        error = error + 1;
        plot(testInputs(j, 1), testInputs(j, 2),'rs');hold on;
    else
        if round(p(j)) == 1
            plot1 = plot(testInputs(j, 1), testInputs(j, 2),'bx');hold on;
        elseif round(p(j)) == 0
            plot2 = plot(testInputs(j, 1), testInputs(j, 2),'go');hold on;
        end
    end
end
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('Logistic fitting model for generic2 test data')
hold on;
disp(error/total);
%%
% for training data
alpha = 0.25;
total = size(trainingTargets, 1);
figure();
[w, b]=learnLogReg(c1_train, c2_train, alpha);
p = logistic(trainingInputs', w, b);
for j= 1:total
    if round(p(j))~=trainingTargets(j, 1)
        plot(trainingInputs(j, 1), trainingInputs(j, 2),'rs');hold on;
    else
        if round(p(j)) == 1
            plot1 = plot(trainingInputs(j, 1), trainingInputs(j, 2),'bx');hold on;
        elseif  round(p(j)) == 0
            plot2 = plot(trainingInputs(j, 1), trainingInputs(j, 2),'go');hold on;
        end
    end
end
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('Logistic fitting model for generic2 train data')
hold on;

%%
% KNN
% since some ks has the same test error value, choose the 
% smallest k by occam's razor

k=3;
total = size(testTargets, 1);
figure();

 for i = 1: total
     class = knnClassify(testInputs(i,:), k, trainingInputs, trainingTargets);
     disp(class);
     if class ~= testTargets(i,:)
        plot(testInputs(1,i), testInputs(2,i),'rs');hold on;
     else
        if class(1,1) == 1
            plot1=plot(testInputs(i, 1), testInputs(i, 2),'bx');hold on;
        elseif class(1,1) == 0
            plot2=plot(testInputs(i, 1), testInputs(i, 2),'go');hold on;
        end
    end
 end
 
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('KNN fitting model for generic2 test data')
hold on;

%%
k=3;
total = size(trainingTargets, 1);
figure();
 for i = 1: total
     class = knnClassify(trainingInputs(i,:), k, trainingInputs, trainingTargets);
     if class ~= testTargets(i,:)
        plot(trainingInputs(i, 1), trainingInputs(i, 2),'rs');hold on;
     else
        if class(1,1) == 1
            plot1 = plot(trainingInputs(i, 1), trainingInputs(i, 2),'bx');hold on;
        elseif class(1,1) == 0
            plot2 = plot(trainingInputs(i, 1), trainingInputs(i, 2),'go');hold on;
        end
    end
 end
 
xlabel('x')
ylabel('y')
legend([plot1,plot2],'C1','C2');
title('KNN fitting model for generic2 train data')
hold on;
%% Apply 3 Algorithms to Fruit
load('fruit_train');
load('fruit_test');

c1Index = find(target_train(1,:)==1);
c2Index = find(target_train(2,:)==1);

%seperate training data by their class 1 or 2
c1_train = inputs_train(:, c1Index);
c2_train = inputs_train(:, c2Index);

fruKnn = knnStarter(inputs_train', target_train', inputs_test', target_test');
fruLog = logisticStarter(c1_train, c2_train, inputs_test, target_test);
[fruEstimatedTargetGcc, fruErrorGcc] = gccStarter(c1_train, c2_train, inputs_test, target_test);
%% Apply 2 Algorithms to Digits
load('mnist_train');
load('mnist_test');

c1Index = find(target_train(1,:)==1);
c2Index = find(target_train(2,:)==1);

c1_train = inputs_train(:, c1Index);
c2_train = inputs_train(:, c2Index);

mniKnn = knnStarter(inputs_train', target_train', inputs_test', target_test');

mniLog = logisticStarter(c1_train, c2_train, inputs_test, target_test);
%% Plots

% For KNN, plot the test error as a function of k. Use k = 3, 5, 7, ..., 21
figure();
plot(g1Knn(:,1), g1Knn(:,2), '--bs');
hold on;
plot(g2Knn(:,1), g2Knn(:,2), '--go');
legend('generic1', 'generic2');
xlabel('k');
ylabel('test error')
title('Test error for generic1/2 as a func of k by KNN algorithm')
hold on;

% For LR, plot of the test error as a function of the regularization weight. 
% Use alpha = 0.25, 0.5, 1, 2, and 5
figure();
plot(g1Log(:,1), g1Log(:,2), '--bs');
hold on;
plot(g2Log(:,1), g2Log(:,2), '--go');
legend('generic1', 'generic2');
xlabel('k');
ylabel('test error')
title('Test error for generic1/2 as a func of the regularization weight by LR algorithm')
hold on;

% For one instance of KNN, one instance of LR, and the GCC model, 
% plot the model fit to the data. 

%%%%%%%%%%%%%%%%%%%%%%% I choose k=3 for both generic1 and generic2 because
%%%%%%%%%%%%%%%%%%%%%%% the minimum test error value occurs at k=3 choose
%%%%%%%%%%%%%%%%%%%%%%% alpha=0.25 since the plot for different alpha is a
%%%%%%%%%%%%%%%%%%%%%%% horizontal line which means the test error won't
%%%%%%%%%%%%%%%%%%%%%%% change if the value of alpha change


%% Plot test error rate for 3 algorithms for fruit
figure();
plot(fruKnn(:,1), fruKnn(:,2), '--bs');
hold on;
plot(fruLog(:,1), fruLog(:,2), '--go');
hold on;

xAxis = [1; 5; 10; 15; 20];
yAxis = [fruErrorGcc; fruErrorGcc; fruErrorGcc; fruErrorGcc; fruErrorGcc];
plot(xAxis, yAxis, '--ro');
legend('Knn', 'Logistic', 'GCC');
xlabel('parameter');
ylabel('test error')
title('Test error for fruit data by 3 algorithms')
hold on;
%% Plot test error rate for 2 algorithms for mnist
figure();
plot(mniKnn(:,1), mniKnn(:,2), '--bs');
hold on;
plot(mniLog(:,1), mniLog(:,2), '--go');
hold on;

legend('Knn', 'Logistic');
xlabel('parameter');
ylabel('test error')
title('Test error for mnist data by 2 algorithms')
hold on;