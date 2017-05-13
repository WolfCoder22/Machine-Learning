function grad = LogisticReg_gradient(Xtrain, Ytrain, theta)
% Compute the gradient of the log likelihood at theta

% INPUT
%  Xtrain  : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%            already contains the constant feature set to 1)
%  Ytrain  : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  theta   : [n x 1] vector, the current model parameters

% OUTPUT
%  grad    : [n x 1] vector, the gradient of the log likelihood at theta

[m,n]= size(Xtrain);
grad=(1:n)';

[~, pred_Y] = LogisticReg_predict(Xtrain, theta);

%loop through all thetas
for t=1:n
    
    gradI=0;
    
    %loop through all input values
    for i=1:m
        gradI= gradI+ (Ytrain(i)-pred_Y(i))*Xtrain(i,t);
    end
    
    grad(t)=gradI;
end
   
end
