
%%%%%%%%%%%%%%%%%% FIt 
load('a1TrainingData.mat');
plot(x, y, '*b'); 

lossTotal = zeros(12, 1);
xPrime = [-2.2:0.1:2.2];
wMatrix = zeros(13, 12);

for K = 1:12
   % find the LS estimate for the polynomial coefficients
   w = polynomialRegression(K,x,y);
    wMatrix( 1: K+1, K) = w;
    
   % compute the total amount of residual error
   predictedY = evalPolynomial(x,w);
   squaredDifference = (y-predictedY).^2;
   loss = sum(squaredDifference);
   lossTotal(K, 1) = loss;
   
   % plot the fitted model for a new x
   yPrime = evalPolynomial(xPrime,w);
   figure();
   plot(xPrime, yPrime);
   hold on
end

% plot the total error as a function of K
K=[1:1:12];
figure();
plot(K, lossTotal);
title('Total Error as a function of K for training data.')
xlabel('K') 
ylabel('total error')
hold on



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test Models
load('a1TestData.mat');

testLossTotal = zeros(12, 1);

for K = 1:12
  predictedYTest =  evalPolynomial(xTest,wMatrix(:, K));
  % compute the total amount of residual error
  squaredDifference = (yTest-predictedYTest).^2;
  loss = sum(squaredDifference);
  testLossTotal(K, 1) = loss;
 
end

% plot the total error as a function of K
K=[1:1:12];
figure();
plot(K, testLossTotal);
title('Total Error as a function of K for testing data.')
xlabel('K') 
ylabel('total error')
hold on
