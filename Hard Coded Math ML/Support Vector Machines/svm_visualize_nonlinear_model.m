function svm_visualize_nonlinear_model(X, Y, svs, alphas, mode)
% Visualize the model of a linear SVM, given the parameters of the primal.
%
% INPUT:
%  X: [m x n] matrix, where each row is a d-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct output value for the i-th input example. 
%  svs: [nsv x 1] the indices of the training examples that are support vectors
%  alphas: [nsv x 1] the alpha coefficients associated to the support vectors
%  mode: the type of kernel; it is a string that can be either 'linear' or 'polynomial'

% ******************************************************************
% ****************** DO NOT EDIT THIS FUNCTION *********************
% ******************************************************************

% calculate the bias from the learned model
bias = svm_calculate_bias_dual(X, Y, svs, alphas, mode);

positive_idx = find(Y==1);
negative_idx = find(Y==-1);
plot(X(positive_idx,1), X(positive_idx,2), 'ro', 'LineWidth', 2);
hold on;
plot(X(negative_idx,1), X(negative_idx,2), 'bx', 'LineWidth', 2);
xlabel('x_1');
ylabel('x_2');
legend('positive examples', 'negative examples');

x1 = get(gca,'XLim');
x2 = get(gca,'YLim');
x1_idx = x1(1):0.01:x1(2);
x2_idx = x2(1):0.01:x2(2);
des_map = zeros(size(x1_idx,2),size(x2_idx,2));
for i=1:size(x1_idx,2)
    for j=1:size(x2_idx,2)
        xtest = [x1_idx(i), x2_idx(j)];
        [~, des_map(i,j)] = svm_predict_dual(xtest, X, Y, svs, alphas, bias, mode);
    end
end

curve = [];
for i=1:size(x1_idx,2)
    [m idx] = min(abs(des_map(i,:)));
    if m<=10^-2
        curve = [curve; x1_idx(i) x2_idx(idx)];
    end
end
plot(curve(:,1), curve(:,2), 'g-', 'LineWidth', 2);

curve = [];
for i=1:size(x1_idx,2)
    [m idx] = min(abs(des_map(i,:)-1));
    if m<=10^-2
        curve = [curve; x1_idx(i) x2_idx(idx)];
    end
end
plot(curve(:,1), curve(:,2), 'c--', 'LineWidth', 2);

curve = [];
for i=1:size(x1_idx,2)
    [m idx] = min(abs(des_map(i,:)+1));
    if m<=10^-2
        curve = [curve; x1_idx(i) x2_idx(idx)];
    end
end
plot(curve(:,1), curve(:,2), 'c--', 'LineWidth', 2);

end