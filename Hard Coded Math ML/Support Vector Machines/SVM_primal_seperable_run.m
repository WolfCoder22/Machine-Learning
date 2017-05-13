function SVM_primal_seperable_run()
% This script requires the following functions to be implemented:
% svm_train_primal_separable

% Make sure you have followed the given directions (Note: do not remove this line of code)

% Load data
S = load('iris_subset.mat');
X = S.trainsetX;
Y = S.trainsetY;
clear S

% train SVM, and report the hyperplane coefficients, bias, and number of SVs
[w, b, svs] = svm_train_primal_separable(X, Y);
fprintf(1,'w = [%f; %f]; b=%f\n', w(1), w(2), b);
fprintf(1,'Number of SVs: %d\n', numel(svs));

% visualize the trained SVM model: training data, decision boundary, margins, support vectors.
figure;
svm_visualize_linear_model(X, Y, w, b, svs);

% save the plot (Note: do not remove this line of code)
saveas(gcf, 'q4c.fig');

end