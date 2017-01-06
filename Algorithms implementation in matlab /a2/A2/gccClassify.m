
function class = gccClassify(x, p1, m1, m2, C1, C2)
% 
% Inputs
%   x - test examplar
%   p1 - prior probability for class 1
%   m1 - mean of Gaussian measurement likelihood for class 1
%   m2 - mean of Gaussian measurement likelihood for class 2
%   C1 - covariance of Gaussian measurement likelihood for class 1
%   C2 - covariance of Gaussian measurement likelihood for class 2
%
% Outputs
%   class - sgn(a(x)) (ie sign of decision function a(x))


% YOUR CODE GOES HERE.
C1_inverse = inv(C1);
C2_inverse = inv(C2);
aX=-0.5*transpose(x-m1)*C1_inverse*(x-m1)-0.5*log(det(C1))+0.5*transpose(x-m2)*C2_inverse*(x-m2)+0.5*log(det(C2));

if aX>0
    class = [1 0];
else
    class = [0 1];
end

end