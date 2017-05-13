function linearReg_run()
% dependent functions
% linearReg_features
% linearReg_mse
% linearReg_train
% linearReg_predict
% linearReg_cross_validation_error
% linearReg_test_error


% Load the data.
S = load('autompg.mat');
X = S.trainsetX;
Y = S.trainsetY;
Xtest = S.testsetX;
Ytest = S.testsetY;
clear S

% Try different lambda with the linear model.
lambda = 10.^[-5:2:7];
error_cross_validation = linearReg_cross_validation_error(X, Y, lambda, 'linear', 10);
error_test = linearReg_test_error(X, Y, Xtest, Ytest, lambda, 'linear');
figure;
subplot(2,1,1);
plot(log10(lambda), error_cross_validation, '-db');
hold on; 
plot(log10(lambda), error_test, '-*r');
legend('cross validation error', 'test error');
ylabel('squared error per sample');
title('regularized least square with b^l(i)');
xlabel('log_{10}\lambda');

% Try different lambda with the quadratic model.
lambda = 10.^[-5:2:7];
error_cross_validation = linearReg_cross_validation_error(X, Y, lambda, 'quadratic', 10);
error_test = linearReg_test_error(X, Y, Xtest, Ytest, lambda, 'quadratic');
subplot(2,1,2);
plot(log10(lambda), error_cross_validation, '-db');
hold on; 
plot(log10(lambda), error_test, '-*r');
legend('cross validation error', 'test error');
ylabel('squared error per sample');
title('regularized least square with b^q(i)');
xlabel('log_{10}\lambda');

saveas(gcf, 'q5f.fig');

end
