function theta = linearReg_train(X, Y, lamda, mode)
% Trains the regularized least squares regression model using the closed form solution given the training data X, Y.
%
% INPUT:
%  X: [m x d] matrix, where each row is a d-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct output value for the i-th input example. 
%  lambda: regularization hyperparameter (scalar value)
%  mode: type of features, either 'linear' or 'quadratic'. 
%
% OUTPUT:
%  theta: [n x 1] vector, containing the learned model parameters.
%

%get feature vectors for each example based off chosen mode
X2=linearReg_features(X, mode);

[m,n]=size(X2);

%get B and U
B= X2;
U=eye(n); 
U(1,1)=0;

%get new matrix for left side of equation and demensions
newMatrix= (B'*B)+lamda*U;

%fprintf('X2=[%d, %d]\n B Transpose= [%d, %d]\n Y= [%d, %d]', q,w,e,o, t, f);
theta= newMatrix\B'*Y;


end