function [w, b, svs] = svm_train_primal_separable(X, Y)
% Trains a linear SVM (primal formulation, separable case), given the training set.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct label for the i-th input example. 
%
% OUTPUT:
%  w: [n x 1] vector, containing the learned model parameters (coefficients of the hyperplane).
%  b: [1 x 1] scalar value, indicating the bias of the hyperplane.
%  svs: [nsv x 1] the indices of the nsv training examples that are support vectors; 
%       for example, svs(1) will give the index of the first support vector
%       and will be an integer number between 1 and m.

[m, n]= size(X);
n=n+1;


%create matrixs for quadprog input
H= eye(n);
H(1,1)=0;

f= zeros(n,1)';

A(1:m, 1:n)=1;
A(1:m, 2:n)= X;

%mutiple each row by corresponding y values
for v=1:m
    A(v,:)= A(v,:)*(Y(v));
end
A=A*-1;

b= ones(1,m);
b= b*-1;

%comput w and b
z=quadprog(H,f,A,b);

%set w and b values
b=z(1);

w= (1:n-1);
for i=2:n
    w(i-1)= z(i);  
end

index=0;
for j=1:m  
    Xi=X(j,:);
    %check for svs
    if(round(Y(j)*(w*Xi'+b), 6)==1)
        fprintf('index %d= %d\n',j,round(Y(j)*(w*Xi'+b), 6))
        index=index+1;
        svs(index)=j;
    end
end
w=w';
svs=svs';

end