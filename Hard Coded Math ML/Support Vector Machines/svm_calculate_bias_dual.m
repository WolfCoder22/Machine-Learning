function bias = svm_calculate_bias_dual(X, Y, svs, alphas, mode)
% Calculate the bias given the learned parameters of a SVM dual formulation, and its training set.
%
% INPUT:
%  X: [m x n] matrix, where each row is a d-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct label for the i-th input example
%  svs: [nsv x 1] the indices of the training examples that are support vectors
%  alphas: [nsv x 1] the alpha coefficients associated to the support vectors
%  mode: the type of kernel; it is a string that can be either 'linear' or 'polynomial'
%
% OUTPUT:
%  bias: [1 x 1] scalar value containing the bias term of the model

[Na, ~]=size(alphas);

bias=0;
for j=1:Na
    
    %make sure alpha greater than t
    bias=bias+Y(svs(j));
    
    bsum=0;
    %loop  all inputs again
    for p=1:Na
        
        k=svm_kernel(X(svs(p),:), X(svs(j),:), mode);
        bsum= bsum+ Y(svs(p))*k*alphas(p);
    end
    
    bias=bias-bsum;
    
end

bias=bias/Na;
end