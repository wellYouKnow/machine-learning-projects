function y = evalPolynomial(x,w)
% Evaluate the polynomial defined by the given weight vector at the
% given values of x.  This function should work even if x is a vector.

y = zeros(length(x), 1);
for i = 1:length(x)
   for j = 1:length(w)
    y(i, 1) = y(i, 1) + w(j)*x(i)^(j-1);
   end
end
end
