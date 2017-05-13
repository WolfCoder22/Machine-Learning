function svm_dual_with_kernel_run()
% This script requires the following functions to be implemented:
% svm_kernel
% svm_calculate_bias_dual
% svm_train_dual
% svm_predict_dual

% Load the data
S = load('iris_subset.mat');
X = S.trainsetX;
Y = S.trainsetY;
clear S

% train the SVM model (linear case)
mode = 'linear';
C = 100;
[svs, alphas] = svm_train_dual(X, Y, C, mode);
fprintf(1,'Linear model. Number of SVs: %d\n', numel(svs));
% plot
figure;
subplot(1, 2, 1);
svm_visualize_nonlinear_model(X, Y, svs, alphas, mode);
title('Linear model');

% train the SVM model (polynomial case)
mode = 'polynomial';
C = 100;
[svs, alphas] = svm_train_dual(X, Y, C, mode);
fprintf(1,'Polynomial model. Number of SVs: %d\n', numel(svs));
% plot
subplot(1, 2, 2);
svm_visualize_nonlinear_model(X, Y, svs, alphas, mode);
title('Polynomial model');

% save the plot (Note: do not remove this line of code)
saveas(gcf, 'q4h.fig');

end

