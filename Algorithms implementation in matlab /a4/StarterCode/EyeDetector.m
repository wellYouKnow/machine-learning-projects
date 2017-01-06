% Simple Eigen-eyes detector. Load data (thanks to Francisco Estada  
% and Allan Jepson for allowing us to use this dataset).

load trainSet
load testSet

% the variables defined in the .mat files are:
% eyeIm - a 500 x n array, each COLUMN contains a vector that
%         represents an eye image
% nonIm - a 500 x m array, each COLUMN contains a vector that
%	  represents a non-eye image
% sizeIm - size of the eye and non eye images [y x]
who

% Normalize brightness to [0 1]
eyeIm=eyeIm/255;
nonIm=nonIm/255;
testEyeIm=testEyeIm/255;
testNonIm=testNonIm/255;

% You can display images from eyeIm or nonIm using;
%
% imagesc(reshape(eyeIm(:,1),sizeIm));axis image;colormap(gray)
%  - where of course you would select any column

% We will first see how far we can get with classification
% on the original data using kNN. The task is to distinguish
% eyes from non-eyes. This is useful to gain insight about
% how hard this problem is, and how much we can improve
% or lose by doing dimensionality reduction.

% Generate training and testing sets with classes for kNN,
% we need eye images to be on ROWS, not COLUMNS, and we also 
% need a vector with class labels for each

trainSet=[eyeIm'
          nonIm'];
trainClass=[zeros(size(eyeIm,2),1)
            ones(size(nonIm,2),1)];

testSet=[testEyeIm'
         testNonIm'];
testClass=[zeros(size(testEyeIm,2),1)
            ones(size(testNonIm,2),1)];

% Compute matrix of pairwise distances (this takes a while...)
d=som_eucdist2(testSet,trainSet);

% Compute kNN results, I simply chose a reasonable value
% for K but feel free to change it and play with it...
K=5;
[C,P]=knn(d,trainClass,K);

% Compute the class from C (we have 0s and 1s so it is easy)
class=sum(C,2);	  		% Add how many 1s there are
class= (class>(K/2));   % Set to 1 if there are more than K/2
				        % ones. Otherwise it's zero

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes

fprintf(2,'Correct classification rate:\n');
correctEye_knn=length(find(class(1:size(testEyeIm,2))==0))/size(testEyeIm,2)
fprintf(2,'False positive rate:\n');
falseEye_knn=length(find(class(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2)

% Keep in mind the above figures! (and the kNN process, you'll
% have to do it again on the dimension-reduced data later on.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% PCA PART: Your task begins here!
% Do PCA on eyes and non-eyes to generate models for recognition


%%% TO DO:
% First, compute the mean over the eye and non-eye images
% i.e. compute the mean eye image, and the mean non-eye image
eyeMean = mean(eyeIm, 2);
nonMean = mean(nonIm, 2);
%%% TO PRINT:
% Plot the mean eye and mean non-eye images and hand the
% printouts in with your report.
figure(1);
imagesc(reshape(eyeMean, sizeIm));
colormap(gray);
title('Mean eye image');
figure(2);
imagesc(reshape(nonMean, sizeIm));
colormap(gray);
title('Mean non eye image');
%%% TO DO:
% Now, perform the PCA computation as discussed in lecture over the 
% training set. You will do this separately for eye images and non-eye 
% images. This will produce a set of eigenvectors that represent eye 
% images, and a different set of eigenvectors for non-eye images.

% substract mean from the training images
substractedEye = eyeIm - (eyeMean * ones(1, size(eyeIm, 2)));
substractedNon  = nonIm - (nonMean * ones(1, size(nonIm, 2)));
% calculate 2 covariance matrices
eyeCov = cov(substractedEye');
nonCov = cov(substractedNon');
% calculate the eigenvectors and eigenvalues
[eyeVec, eyeD] = eig(eyeCov);
[noneyeVec, nonD] = eig(nonCov);

[c1, ind] = sort(diag(eyeD), 'descend');
eyeVec = eyeVec(:, ind);

[c2, ind] = sort(diag(nonD), 'descend');
noneyeVec = noneyeVec(:, ind);
%%% TO PRINT:
% Display and print out the first 5 eigenvectors for eyes and non-eyes 
% (i.e. the eigenvectors with LARGEST 5 eigenvalues, make sure you sort 
% the eigenvectors by eigenvalues!)
figure(3);
for i = 1:5
    subplot(2, 5, i);
    imagesc(reshape(eyeVec(:, i), sizeIm));
    axis image;
    colormap(gray);
    hold on;
    
    subplot(2, 5, i+5);
    imagesc(reshape(noneyeVec(:, i), sizeIm));
    axis image;
    colormap(gray);
    hold on;
end
%%% TO DO:
% Now you have two PCA models: one for eyes, and one for non-eyes. 
% Next we will project our TEST data onto our eigenmodels to obtain 
% a low-dimensional representation first we choose a number of PCA
% basis vectors (eigenvectors) to use:

% PCAcomp=10;	% Choose 10 to start with, but you will 
		% experiment with different values of 
		% this parameter and see how things work
% correct classification rate for PCA      
cPCA = zeros(1, 5);
% false positive rate for PCA
fPCA = zeros(1, 5);

% for KNN
cKNN = zeros(1, 5);
fKNN = zeros(1, 5);

count = 0;
for PCAcomp = [5, 10, 15, 25, 50]
% To compute the low-dimensional representation for a given
% entry in the test test, we must do 2 things. First, we subtract
% the mean, and then we project that vector on the transpose of 
% the PCA basis vectors.  For example, say you have an eye image
%
% vEye=testSet(1,:);   % This is a 1x500 row vector
%
% The projections onto the PCA eigenvectors are:
%
% coeffEye=eyeVec(:,1:PCAcomp)'*(vEye'-eyeMean);
% coeffNonEye=noneyeVec(:,1:PCAcomp)'*(vNonEye'-noneyeMean);
%
% You need to compute coefficients for BOTH the eye and non-eye 
% models for each testSet entry, i.e. for each testSet image you 
% will end up with (2*PCAcomp) coefficients which are the projection 
% of that test image onto the chosen eigenvectors for eyes and non-eyes.
    count = count + 1;
    substractedTestEye = testSet' - (eyeMean * ones(1, size(testSet, 1)));
    substractedTestNon  = testSet' - (nonMean * ones(1, size(testSet, 1)));
    coeffEye = substractedTestEye' * eyeVec(:, 1:PCAcomp);
    coeffNonEye = substractedTestNon' * noneyeVec(:, 1:PCAcomp);
% Since we are going to use the KNN classifier demonstrated above, 
% you might want to place all the of the test coefficients into one 
% matrix.  You would then end up with a matrix that has one ROW for 
% each image in the testSet, and (2*PCAcomp) COLUMNS, one for each 
% of the coefficients we computed above.
    coeffs = [coeffEye coeffNonEye];

%%% TO DO:
% Then do the same for the training data.  That is, compute the 
% PCA coefficients for each training image using both of the models.
% Then you will have low-dimensional test data and training data
% ready for the application of KNN, just as we had in the KNN example
% at the beginning of this script.

    substractedTrainEye = trainSet' - (eyeMean * ones(1, size(trainSet, 1)));
    substractedTrainNon  = trainSet' - (nonMean * ones(1, size(trainSet, 1)));
    coeffTrainEye = substractedTrainEye' * eyeVec(:, 1:PCAcomp);
    coeffTrainNonEye = substractedTrainNon' * noneyeVec(:, 1:PCAcomp);
    coeffsTrain = [coeffTrainEye coeffTrainNonEye];

%%% TO DO
% KNN classification: 
% Repeat the procedure at the beginning of this script, except
% instead of using the original testSet data, use the 
% coefficients for the training and testing data, and the same
% class labels for the training data that we had before
%

    dprime=som_eucdist2(coeffs,coeffsTrain);

% Compute kNN results, I simply chose a reasonable value
% for K but feel free to change it and play with it...
    K = 5;
    [Cprime,Pprime]=knn(dprime,trainClass,K);

% Compute the class from C (we have 0s and 1s so it is easy)
    class=sum(Cprime,2);	  		% Add how many 1s there are
    class= (class>(K/2));   % Set to 1 if there are more than K/2
				        % ones. Otherwise it's zero


%%% TO PRINT:
% Print the classification accuracy and false-positive rates for the
% kNN classification on low-dimensional data and compare with the
% results on high-dimensional data.
%
% Discuss in your report: 
% - Are the results better? worse? is this what you expected?
% - why do you think the results are like this?
%

% Compute classification accuracy: We're interested in 2 numbers:
% Correct classification rate - how many eyes were classified as eyes
% False-positive rate: how many non-eyes were classified as eyes
    correctEye_knn_low_d = length(find(class(1:size(testEyeIm,2))==0))/size(testEyeIm,2);
    falseEye_knn_low_d = length(find(class(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2);
    cKNN(count) = correctEye_knn_low_d;
    fKNN(count) = falseEye_knn_low_d;
    fprintf(2,'Correct classification rate at PCAcomp = %d:\n', PCAcomp);
    disp(correctEye_knn_low_d)
    fprintf(2,'False positive rate at PCAcomp = %d :\n', PCAcomp);
    disp(falseEye_knn_low_d);

%%% TO DO:
% Finally, we will do classification directly from the PCA models
% for eyes and non-eyes.
%
% The idea is simple: Reconstruct each entry in the testSet
% using the PCA model for eyes, and separately the PCA model
% for non-eyes. Compute the squared error between the original
% entry and the reconstructed versions, and select the class
% for which the reconstruction error is smaller. It is assumed
% that the PCA model for eyes will do a better job of
% reconstructing eyes and the PCA model for non-eyes will
% do a better job for non-eyes (but keep this in mind:
% there's much more stuff out there that is not an eye
% than there are eyes!)
%
% To do the reconstruction, let's look at a vector from the
% coefficients we computed earlier for the training set;
%
% Reconstruction
%
% vRecon_eye= eyeMean + sum_k (eye_coeff_k * eye_PCA_vector_k);
%
% i.e. the mean eye image, plus the sum of each PCA component 
% multiplied by the corresponding coefficient. One can also replace
% the sum with a matrix-vector product.  Note: If you don't add 
% the mean image component back this won't work!
%
% Likewise, for the reconstruction using the non-eye model
%
% vRecon_noneye= nonMean + sum_k (noneye_coeff_k * noneye_PCA_vector_k)
%

%%% TO DO:
%
% Compute the reconstruction for each entry using the PCA model for eyes
% and separately for non-eyes, compute the error between these 2 
% reconstructions and the original testSet entry, and select the class
% that yields the smallest error.
%
    recEye = eyeMean * ones(1, size(testSet, 1)) + eyeVec(:, 1:PCAcomp) * coeffEye';
    recNonEye = nonMean * ones(1, size(testSet, 1)) + noneyeVec(:, 1:PCAcomp) * coeffNonEye';
    label = zeros(size(testSet, 1), 1);
    for i = 1: size(testSet, 1)
        distEye = norm(recEye(:, i)-testSet(i, :)');
        disNonEye = norm(recNonEye(:, i)-testSet(i, :)');
        label(i) = disNonEye < distEye;
    end
    correctEye_PCA = length(find(label(1:size(testEyeIm,2))==0))/size(testEyeIm, 2);
    falseEye_PCA = length(find(label(size(testEyeIm,2)+1:end)==0))/size(testNonIm,2);
    cPCA(count) = correctEye_PCA;
    fPCA(count) = falseEye_PCA;
end
%%% TO PRINT:
%
% Print the correct classification rate and false positive rate for
% the PCA based classifier and the low-dimensional kNN classifier
% using PCAcomps=5,10,15,25, and 50
disp('PCA correct classification rate for PCAcomp = 5, 10, 15, 25, 50: ')
disp(cPCA);
disp('PCA false positive rate for PCAcomp = 5, 10, 15, 25, 50: ')
disp(fPCA);
disp('low-dimensional KNN correct classification rate for PCAcomp = 5, 10, 15, 25, 50: ')
disp(cKNN);
disp('low-dimensional KNN false positive rate for PCAcomp = 5, 10, 15, 25, 50: ')
disp(fKNN);
% Plot a graph of the kNN classification rate for the low-dimensional
% KNN classifier VS the number of PCA components (for the 5 values of 
% PCAcomps requested). 
figure(4);
plot([5, 10, 15, 25, 50], cKNN, 'g--');
hold on;
plot([5, 10, 15, 25, 50], fKNN, 'b--')

legend('correct classification rate', 'false positive rate');
xlabel('number of PCA components(PCAcomp)');
ylabel('classification rate');
title('KNN classification rate for different value of PCAcomp on the low-dimensional data');
% Discuss in your Report:
% - Is there a value for PCAcomps (or set of values) for which low-dimensional
%   kNN is better than full dimensional kNN? 
% - why do you think that is?
%
% Plot graphs of correct classification rate and the false-positive rate 
% fr the PCA-reconstruction classifier vs the number of PCA components.
%
figure(5);
plot([5, 10, 15, 25, 50], cKNN, 'g--');
hold on;
plot([5, 10, 15, 25, 50], fKNN, 'b--');

legend('correct classification rate', 'false positive rate');
xlabel('number of PCA components(PCAcomp)');
ylabel('classification rate');
title('PCA classification rate for different value of PCAcomp on the reconstruction data');

figure(6);
plot([5, 10, 15, 25, 50], cKNN, 'g--');
hold on;
plot([5, 10, 15, 25, 50], cPCA, 'b--');

legend('KNN', 'PCA');
xlabel('number of PCA components(PCAcomp)');
ylabel('correct classification rate');
title('correct classification rates of KNN and PCA classifiers for different value of PCAcomp');

figure(7);
plot([5, 10, 15, 25, 50], fKNN, 'g--');
hold on;
plot([5, 10, 15, 25, 50], fPCA, 'b--');

legend('KNN', 'PCA');
xlabel('number of PCA components(PCAcomp)');
ylabel('false positive rate');
title('false positive rates of KNN and PCA classifiers for different value of PCAcomp');
% Discuss in your Report:
% - Which classifier gives the overall best performance?
% - What conclusions can you draw about the usefulness of dimensionality
%   reduction?
% - Which classifier would you use on a large training set
%   consisting of high-dimensional data?
% - Which classifier would you use on a large training set
%   of low-dimensional (e.g. 3-D) data?
% - why?
% - Summarize the advantages/disadvantages of each classifier!
%
