function SVM_primal_run()
% This script requires the following functions to be implemented:
% svm_train_primal
% svm_predict_primal

% Load data
S = load('iris_subset.mat');
X = S.trainsetX;
Y = S.trainsetY;
Xtest = S.testsetX;
Ytest = S.testsetY;
clear S

% Try a bunch a hyperparameters for "C"
C_list = [0.1, 1, 10, 100];
figure;
for idx_C = 1:numel(C_list)
    C = C_list(idx_C);
    % train the SVM (slack version), for a particular hyperparameter C
    [w, b, svs] = svm_train_primal(X, Y, C);
    
    % visualize the model in a subplot
    subplot(1, numel(C_list), idx_C);
    svm_visualize_linear_model(X, Y, w, b, svs);
    title(['C=' num2str(C)]);
    
    % calculate and print the training and test error
    [pred_train_labels, ~] = svm_predict_primal(X, w, b);
    train_error = sum(Y ~= pred_train_labels)/numel(Y);
    [pred_test_labels, ~] = svm_predict_primal(Xtest, w, b);
    test_error = sum(Ytest ~= pred_test_labels)/numel(Ytest);
    fprintf(1, 'C=%.2f, number of SVs: %d, train error: %.2f%%, test error: %.2f%%\n', ...
    	C, size(svs,1), 100*train_error, 100*test_error);
end

% save the plot (Note: do not remove this line of code)
saveas(gcf, 'q4e.fig');

end

