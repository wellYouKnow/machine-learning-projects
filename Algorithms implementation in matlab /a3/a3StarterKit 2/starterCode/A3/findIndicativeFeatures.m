function [c1_features,c0_features,c1_features_index,c0_features_index]=findIndicativeFeatures(pred_test, data_test, feature_names)

c0_test = data_test(find(pred_test == 0), :);
c1_test = data_test(find(pred_test == 1), :);

% count occurrences of each feature in both classes
counts_features_c0 = sum(c0_test);
counts_features_c1 = sum(c1_test);

% probabilities of each feature in class i
p_c0 = counts_features_c0 ./ size(c0_test, 1);
p_c1 = counts_features_c1 ./ size(c1_test, 1);


%[sorted_c0, index] = sort(p_c0, descend);
%[sorted_c1, index1] = sort(p_c1, descend);

% find 10 features has a high occurance rate in only one class 
% higher diff implies that this feature implies data is more class c0
% than class c1
diff = p_c0 - p_c1;
[sorted, index] = sort(diff);

% 10 most indicative features

% descending indicative of a message being class 1
c1_features_index = index(1:10);
% ascending indicative of a message being class 0
c0_features_index = index(176:185);

% find 10 indicative features' name
c1_features = feature_names(c1_features_index)';
c0_features = feature_names(c0_features_index)';