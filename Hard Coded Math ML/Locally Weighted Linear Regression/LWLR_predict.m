function pred_y = LWLR_predict(X, Y, xtest, tau)
% Predicts the output value of the input example xtest, given the training
% set X, Y, parameter tau, and the test example

% 
% INPUT
%  X     : [m x d] matrix, where each row is a d-dimensional input example
%  Y     : [m x 1] vector, where the i-th element is the correct output value 
%                for the i-th input example. 
%  xtest : [d x 1] vector, the input vector of a *single* test example
%  tau   : scalar, value of the regularization hyperparameter
%
% OUTPUT
%  pred_y: scalar, the predicted output value.
%

theta = LWLR_train(X, Y, xtest, tau);
B=LWLR_features(xtest', 'linear');
    
pred_y= B*theta;

end
