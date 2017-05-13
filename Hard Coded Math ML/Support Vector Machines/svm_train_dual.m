function [svs, alphas] = svm_train_dual(X, Y, C, mode)
% Trains a linear SVM (non-separable dual formulation).
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct label for the i-th input example.
%  C: [1 x 1] scalar value, indicating the hyperparameter "C" to use in the SVM formulation.
%  mode: the type of kernel; it is a string that can be either 'linear' or 'polynomial'
%
% OUTPUT:
%  svs: [nsv x 1] the indices of the nsv training examples that are support vectors; 
%       for example, svs(1) will give the index of the first support vector
%       and will be an integer number between 1 and m.
%  alphas: [nsv x 1] the alpha coefficients associated to the support vectors

t=10^(-5);
[m, ~]= size(X);

%create matrix for quadprog input
H= eye(m);

%loop through all entries twice to make H matrix
for i=1:m
    for r=1:m
        
        k=svm_kernel(X(i,:), X(r,:), mode);
        hEntry= Y(i)*Y(r)*k;
        
        H(i,r)=hEntry;
    end
end

f= ones(m,1)';
f=f*-1;

A=[];
d=[];

Aeq=Y';
beq=0;

ub= ones(1,m)';
ub=ub*C;

lb=zeros(1,m)';

%comput alpha coffiecnts
a=quadprog(H,f,A,d,Aeq,beq,lb,ub);


%loop pver al alphas
index=0;
for j=1:m  
    
    %check for svs
    if(a(j)>=t)
        index=index+1;
        svs(index)=j;
        
        %store alpha value
        alphas(index)=a(j);
    end
end

alphas=alphas';
svs=svs';
end