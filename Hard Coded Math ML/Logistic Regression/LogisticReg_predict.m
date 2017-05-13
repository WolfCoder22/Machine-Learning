function [pred_Y, prob_Y] = LogisticReg_predict(X, theta)
% Predict the labels and probabilities for the set of examples X using the
% model theta

% INPUT
%  X      : [m x n] matrix, where each row is an n-dimensional input example (please assume it 
%            already contains the constant feature set to 1)
%  theta  : [n x 1] vector, the model parameters used to make predictions

% OUTPUT
%  pred_Y : [m x 1] vector, the predicted labels for the examples in X
%                   note that pred_Y has binary values {0,1} in this case
%  prob_Y : [m x 1] vector, the posterior probabilities produced by the logistic function

[m,~] =size(X);

%initialize output
pred_Y= (1:m)';
prob_Y= (1:m)';

%get z values
z=X*theta;

for i=1:m
    
    g= exp(z(i))/(1+exp(z(i)));
    prob_Y(i)=g;
    
    if(g<=.5)
        pred_Y(i)=0;
    else
        pred_Y(i)=1;
    end
end

end
