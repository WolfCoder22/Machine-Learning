function svm_visualize_linear_model(X, Y, w, b, svs)
% Visualize the model of a linear SVM, given the parameters of the primal.
%
% INPUT:
%  X: [m x n] matrix, where each row is a d-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct output value for the i-th input example. 
%  w: [n x 1] vector, containing the learned model parameters (coefficients of the hyperplane).
%  b: [1 x 1] scalar value, indicating the bias of the hyperplane.
%  svs: [nsv x 1] the indices of the training examples that are support vectors.

% ******************************************************************
% ****************** DO NOT EDIT THIS FUNCTION *********************
% ******************************************************************

% visualize the training data
positive_idx = find(Y==1);
negative_idx = find(Y==-1);
plot(X(positive_idx,1), X(positive_idx,2), 'ro', 'LineWidth', 2);
hold on;
plot(X(negative_idx,1), X(negative_idx,2), 'bx', 'LineWidth', 2);
legend('positive examples', 'negative examples');

% visualize the decision boundary, and the pos/neg margin
draw_line(w, b, 'k-');
draw_line(w, b+1, 'g--');
draw_line(w, b-1, 'g--');

% draw the Support Vectors
for i=1:numel(svs)
    if Y(svs(i))==1
        plot(X(svs(i),1), X(svs(i),2), 'mo', 'LineWidth', 2);
    else
        plot(X(svs(i),1), X(svs(i),2), 'mx', 'LineWidth', 2);
    end
end
end


function draw_line(w, b, style)

assert(numel(b) == 1);
assert(numel(w) == 2);

z = [b ; w];
x = get(gca, 'XLim');
y = (-z(2)*x-z(1))./z(3);
plot(x, y, style,'LineWidth', 2);

end