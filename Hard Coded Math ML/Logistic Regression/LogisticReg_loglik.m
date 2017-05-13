function lik = LogisticReg_loglik(Xtrain, Ytrain, theta)
% Computes the log likelihood value for training data (Xtrain, Ytrain) and parameter theta

% INPUT
%  Xtrain  : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%            already contains the constant feature set to 1)
%  Ytrain  : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  theta   : [n x 1] vector, the current model parameters

% OUTPUT
%  lik     : [1 x 1] scalar, the computed log likelihood

m= size(Xtrain, 1);
% lik=0;
[~, prob_y] = LogisticReg_predict(Xtrain, theta);
% 
% for i=1:m
%      lik= lik + Ytrain(i)*log(prob_y(i)+eps)+ (1-Ytrain(i))*log(1-prob_y(i)+eps);
% end

lik= sum(Ytrain.*log(prob_y+eps)+(1-Ytrain).*log(-1*prob_y+1+eps));
end
