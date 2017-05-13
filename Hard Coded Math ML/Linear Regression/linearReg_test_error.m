function error = linearReg_test_error(X, Y, Xtest, Ytest, lamda, mode)
% Given training and test set, it trains the model and calculates the test error.
%
% INPUT
%  X: [m x d] matrix, where each row is a d-dimensional training example
%  Y: [m x 1] vector, where the i-th element is the output value 
%     for the i-th training example.
%  Xtest: [M x d] matrix, where each row is a d-dimensional test example
%  Ytest: [M x 1] vector, containing the output values of the test examples
%  lambda: [1 x K] vector, containing the list of reguralization parameters
%  mode: type of features, wither 'linear' or 'quadratic'
%
% OUTPUT
%  error: [1 x K] vector containing test errors, one for each lambda.
%

[~, kNum]= size(lamda);

error=(1:kNum);

%loop through all lamda values
for k=1:kNum
    
    theta = linearReg_train(X, Y, lamda(k), mode);
    pred_Y = linearReg_predict(theta, Xtest, mode);
    err = mse(pred_Y, Ytest);
    
    %assign error to output
    error(1, k)=err;
  
end
end