function D = kmeans_dist2(X1, X2)
% Calculates the *squared* Euclidean distance between two sets of points.
% You should be able to efficiently implement this function *without* using for-loops.
%
% INPUT:
%  X1: [m1 x n] matrix, where each row is an n-dimensional input example
%  X2: [m2 x n] matrix, where each row is an n-dimensional input example
%  
% OUTPUT:
%  D: [m1 x m2] matrix, where the element D(i,j) represent the squared Euclidean distance between
%               the i-th example of X1, and the j-th example of X2.

x = X1;
c = X2;

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
    error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
    ones(ndata, 1) * sum((c.^2)',1) - ...
    2.*(x*(c'));

% Rounding errors occasionally cause negative entries in n2
if any(any(n2<0))
    n2(n2<0) = 0;
end

D = n2;

end