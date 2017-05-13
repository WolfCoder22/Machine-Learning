function [theta, n_iter, loglik] = LogisticReg_train_line_search(Xtrain, Ytrain, theta_init, alpha0, tol)
% Train logistic regression using gradient ascent via line search
% 
% INPUT
%  Xtrain    : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%              already contains the constant feature set to 1)
%  Ytrain    : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  theta_init: [n x 1] vector, the initial parameter vector
%  alpha0    : [1 x 1] scalar, the initial (large) step size used for line search
%  tol       : [1 x 1] scalar, tolerance value used in the stopping condition

% OUTPUT
%  theta   : [n x 1] vector, the learned parameters
%  n_iter  : [1 x 1] scalar, the number of gradient ascent iterations until convergence
%  loglik  : [1 x n_iter] vector containing the log likelihood value at each iteration

% HINTS
%  your program should use the following stopping criterion:
%        while (norm(grad)>tol) && (n_iter < 100000)
%
% where grad is the gradient at the current iteration

grad= LogisticReg_gradient(Xtrain, Ytrain, theta_init);
n_iter=1;
theta= theta_init;
loglik(n_iter)= LogisticReg_loglik(Xtrain, Ytrain, theta);

%stopping criteria for ascent
while (norm(grad)>tol) && (n_iter < 100000)
    
    grad= LogisticReg_gradient(Xtrain, Ytrain, theta);
    a=alpha0;
    n_iter= n_iter +1;
    loglik(n_iter)= LogisticReg_loglik(Xtrain, Ytrain, theta);
    
    %find best step size
    while(LogisticReg_loglik(Xtrain, Ytrain, theta+(a*grad)) <= LogisticReg_loglik(Xtrain, Ytrain, theta));
        a=a/2;
    end
    
    theta= theta+ a*grad;
    
end

end
