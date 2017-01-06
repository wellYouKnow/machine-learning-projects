load('a3spam.mat')
test_data_size = size(data_test, 1);

%%
% Logistic Regression classifier

error = zeros(1, 10);

% first row is 1s, each column is a data vector
X = [ones(1, 1000); data_train'];
i = 1;
for v = 1:1:10
    [beta, converged] = logisticReg(X, labels_train', v);
    p = logistic([ones(1, 4000); data_test'], beta);
    pred_test = p >= 0.5;
    error(1, i) = sum(pred_test~=labels_test');
    i = i + 1;
end

figure(1);clf;
plot(1:1:10, error, '--re');
xlabel('regularization parameter v: variance for a Gaussian prior on the weights');
ylabel('test error');
title('Test Error of Logistic Regression as a function of v');

%%
% when v=4, test error is 0
v = 4;
[beta, converged] = logisticReg(X, labels_train', v);
p = logistic([ones(1, 4000); data_test'], beta);
pred_test = p >= 0.5;

[c1_LR_features, c0_LR_features, c1_features_index, c0_features_index] = findIndicativeFeatures(pred_test, data_test, feature_names);

% find their corresponding weight
c1_LR_weights = beta(c1_features_index)';
c0_LR_weights = beta(c0_features_index)';


%%
% Naive Bayes classifier

error_naive = zeros(9, 9);
i = 1;
for alpha = 0.1:0.05:0.5
    j = 1;
	for beta = 0.1:0.05:0.5
    	p = naiveBayes(data_test, labels_test, data_train, labels_train, alpha, beta);
    	pred_test = p >= 0.5;
    	error_naive(i, j) = sum(pred_test~=labels_test);
    	j = j + 1;
    end
    i = i + 1;
end

% 3d plot
figure(2);clf;
plot3(0.1:0.05:0.5, 0.1:0.05:0.5, error_naive, '--bs');
xlabel('alpha');
ylabel('beta');
zlabel('test error');
title('Test Error of Naive Bayes as a function of alpha and beta');

figure(3);clf;
% 2d plot with beta = 0.1
plot(0.1:0.05:0.5, error_naive(:, 1), '--re');
xlabel('alpha');
%ylabel('beta');
ylabel('test error');
title('Test Error of Naive Bayes as a function of alpha and beta=0.1');

figure(4);clf;
plot(0.1:0.05:0.5, error_naive(1, :), '--bs');
xlabel('beta');
%ylabel('beta');
ylabel('test error');
title('Test Error of Naive Bayes as a function of alpha = 0.1 and beta');
%%
% test error is minimized when alpha = beta = 0.1 
alpha = 0.1;
beta = 0.1;
[prob, a_1, a_0] = naiveBayes(data_test, labels_test, data_train, labels_train, alpha, beta);
pred_test = prob >= 0.5;

[c1_NB_features, c0_NB_features, c1_features_index, c0_features_index] = findIndicativeFeatures(pred_test, data_test, feature_names);

c1_NB_weights = a_1(c1_features_index);
c0_NB_weights = a_0(c0_features_index);





