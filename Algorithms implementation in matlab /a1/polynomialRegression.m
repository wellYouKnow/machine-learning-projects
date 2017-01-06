function w = polynomialRegression(K,x,y)
% Fill in this function.  It should return a (K+1)x1 weight vector w where the
% estimated model function is f(x) = \sum_{i=0}^{K} w(i) x^i.
% for test only
%x = [-2, -1.8, -1.6, -1.4, -1.2, -1];
%K = 5;
%y = [4;1.8^2; 1.6^2; 1.4^2; 1.2^2; 1];
    
%initialize x matrix first
xMatrix = zeros(length(x), K+1);

% assign values to xMatrix
for i = 1:length(x)
   for j = 0:K
    xMatrix(i, j+1) = x(i)^j;
   end
end

w = xMatrix\y;
end
