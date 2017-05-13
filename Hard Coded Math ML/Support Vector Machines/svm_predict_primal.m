function [label, S] = svm_predict_primal(X, w, b)
% Predict the labels of the given examples, given the learned parameters for the 
% primal SVN formulation.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  w: [n x 1] vector, containing the learned model parameters (coefficients of the hyperplane).
%  b: [1 x 1] scalar value, indicating the bias of the hyperplane
%
% OUTPUT:
%  label: [m x 1] vector containing the predicted labels (+1 or -1)
%  S: [m x 1] vector, where the i-th element is the SVM score for the i-th
%     input example, i.e., (w'*x + b) in case of a single example x

[m, ~]= size(X);
label=(1:m);
S=(1:m);

%loop thorugh all x inputs
for i=1:m
    score= w'*X(i,:)' + b;
    S(i)= score;
    
    %get label
    if(score>0)
        label(i)=1;
    else
        label(i)=-1;
    end
end
label=label';
S=S';

end