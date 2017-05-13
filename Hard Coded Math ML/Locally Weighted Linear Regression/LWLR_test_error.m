function error = LWLR_test_error(Xtrain, Ytrain, Xtest, Ytest, tau)
% Given training and test set, it trains the model and calculates the test error.
%
% INPUT
%  Xtrain: [m x d] matrix, where each row is a d-dimensional training example
%  Ytrain: [m x 1] vector, where the i-th element is the output value 
%                  for the i-th training example.
%  Xtest : [M x d] matrix, where each row is a d-dimensional test example
%  Ytest : [M x 1] vector, containing the output values of the test examples
%  tau   : [1 x K] vector, containing the set of values for the reguralization parameter
%
% OUTPUT
%  error : [1 x K] vector, containing test errors, one for each value of tau.
%

[m,~]= size(Xtest);
[~, k]= size(tau);

pred_y= zeros(m,1);
error= zeros(1,k);

%loop through all tau's
for j=1:k;
    
    %go through every X in test set and get pred through LWLR
    for i=1:m
       xtester=Xtest(i,:);
       pred_y(i,1) = LWLR_predict(Xtrain, Ytrain, xtester', tau(j)); 
    end
    
    %set error for specific tau
    error(j)= mse(pred_y, Ytest);
    
    pred_y= zeros(m,1); %reset Y
    
end

end

