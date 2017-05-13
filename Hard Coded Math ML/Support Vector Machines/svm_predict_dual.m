function [label, s] = svm_predict_dual(xtest, Xtrain, Ytrain, svs, alphas, bias, mode)
% Predict the labels of the given examples, given the learned parameters for the 
% primal SVN formulation.
%
% INPUT:
%  xtest: [1 x n] vector which is a single n-dimensional test example
%  Xtrain: [m x n] matrix, where each row is an n-dimensional input example of the training set
%  Ytrain: [m x 1] vector, where the i-th element is the correct label for the i-th training example.
%  svs: [nsv x 1] the indices of the training examples that are support vectors
%  alphas: [nsv x 1] the alpha coefficients associated to the support vectors
%  bias: [1 x 1] scalar value containing the bias term of the model
%  mode: the type of kernel; it is a string that can be either 'linear' or 'polynomial'
%
% OUTPUT:
%  label: [1 x 1] scalar containing the predicted label for the test example (+1 or -1)
%  s: [1 x 1] scalar, the SVM score for the test example

[nsv, ~]= size(svs);
s=0;

%loop  all support vectors
for p=1:nsv

    k=svm_kernel(Xtrain(svs(p),:), xtest, mode);
    s= s+ Ytrain(svs(p))*k*alphas(p);
end
s= s+bias;

%get label
if(s>=0)
    label=1;
else
    label=-1;
end
end
