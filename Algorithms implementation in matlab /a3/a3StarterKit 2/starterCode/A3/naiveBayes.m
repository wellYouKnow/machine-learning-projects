function [prob, a_0, a_1] = naiveBayes(data_test, labels_test, data_train, labels_train, alpha, beta)

% Build a naive Bayes classifier
% Inputs: 
% data_test/data_train: N*M matrix where N is the number of data, 
%                       M is the number of features
% labels_test/labels_train: N*1 matrix where N is the number of data, 
%                           each row is the label for corresponding data
% alpha/beta: regularization parameter
% Outputs:
% prob: probability of data being class 1



train_size = size(data_train, 1);
num_features = size(data_train, 2);

% since labels are 1 and 0
num_c1 = sum(labels_train); 
num_c0 = train_size - num_c1;

% seperate data by different classes
c0_train = data_train(find(labels_train==0), :);
c1_train = data_train(find(labels_train==1), :);

a_0 = zeros(1, num_features);
a_1 = zeros(1, num_features);
% a_{i,k}=(N_{i,k}+alpha)/(N_{k}+2*alph)
for i = 1:num_features
    a_0(1, i) = (sum(c0_train(:, i)) + alpha) / (num_c0 + 2*alpha);
    a_1(1, i) = (sum(c1_train(:, i)) + alpha) / (num_c1 + 2*alpha);  
end

% b_{k}=(N_{k}+beta)/(N + K*beta)
b_1 = (num_c1 + beta) / (train_size + 2*beta);
b_0 = (num_c0 + beta) / (train_size + 2*beta);

% the larger probability of being class 1
prob = zeros(size(labels_test, 1), 1);

for n = 1:size(labels_test, 1)
      a0 = 0;
      a1 = 0;
      for m = 1:num_features
          if data_test(n, m) == 1 % if this feature presents
              a1 = a1 + log(a_1(1, m));
              a0 = a0 + log(a_0(1, m));
          else
              a1 = a1 + log(1 - a_1(1, m));
              a0 = a0 + log(1 - a_0(1, m));              
          end
      end
      a1 = a1 + log(b_1);
      a0 = a0 + log(b_0);
      gamma = max(a1, a0);
      prob(n, 1) = exp(a1 - gamma) / (exp(a1 - gamma) + exp(a0 - gamma));
end
