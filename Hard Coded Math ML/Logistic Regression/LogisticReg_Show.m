function LogisticReg_Show()
% This script requires the following functions to be implemented:
% LogisticReg_initialize
% LogisticReg_predict
% LogisticReg_loglik
% LogisticReg_gradient
% LogisticReg_train
% LogisticReg_error

% load data
S = load('parkinsons.mat');
X = S.trainsetX;
Y = S.trainsetY;
Xtest = S.testsetX;
Ytest = S.testsetY;

clear S;

% add constant feature set to 1 in order to implement the bias term
m = size(X,1);
X = [ones(m,1) X];
m = size(Xtest,1);
Xtest = [ones(m,1) Xtest];


alpha = 1e-6; % learning rate / step size
tol = 6; % tolerance on the norm of the gradient to decide when to stop

% initialize weights
theta_init = LogisticReg_initialize(X, Y, 'heuristic');

[theta, n_iter, loglik] = LogisticReg_train(X, Y, theta_init, alpha, tol);

[pred_Y, ~] = LogisticReg_predict(X, theta);
train_error = LogisticReg_error(Y, pred_Y);

[pred_Ytest, ~] = LogisticReg_predict(Xtest, theta);
test_error = LogisticReg_error(Ytest, pred_Ytest);

fprintf('Number of iterations: %d\n', n_iter);
fprintf('Misclassification rate on the training set: %f%%\n', train_error*100);
fprintf('Misclassification rate on the test set: %f%%\n', test_error*100);

% close figures
close all;

plot(1:n_iter, loglik, 'o-');
ylabel('Log likelihood');
title('Log likelihood as a function of the number of iterations');
xlabel('Number of iterations');
grid on;

saveas(gcf, 'q2a.fig');
