function [w, b, svs] = svm_train_primal(X, Y, C)
% Trains a linear SVM (prima formulation, non-separable case), given a training set.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  Y: [m x 1] vector, where the i-th element is the correct label for the i-th input example.
%  C: [1 x 1] scalar value, indicating the hyperparameter C to use in the SVM formulation.
%
% OUTPUT:
%  w: [n x 1] vector, containing the learned model parameters (coefficients of the hyperplane).
%  b: [1 x 1] scalar value, indicating the bias of the hyperplane
%  svs: [nsv x 1] the indices of the nsv training examples that are support vectors; 
%       for example, svs(1) will give the index of the first support vector
%       and will be an integer number between 1 and m.

opts = struct( ...
    'Algorithm','active-set', ...
    'Diagnostics','off', ...
    'Display','final', ...
    'HessMult',[], ... 
    'MaxIter',[], ...    
    'MaxPCGIter','max(1,floor(numberOfVariables/2))', ...   
    'PrecondBandWidth',0, ... 
    'TolCon',1e-8, ...
    'TolFun',[], ...
    'TolFunValue', [], ...
    'TolPCG',0.1, ...    
    'TolX',100*eps, ...
    'TypicalX','ones(numberOfVariables,1)' ...    
    );

[m, n]= size(X);
n2=n+m+1;

%create matrixs for quadprog input
H= eye(n2);
H(1,1)=0;

%make digonols on n+2-end be 0's
for s=n+2:n2
    H(s,s)=0;
end

%create f vector to cancel out w and b values in z opitimation
f= zeros(n2,1);
for e=n+2:n2
    f(e)=C;
end

A1(1:m, 1:n+1)=1;
A1(1:m, 2:n+1)= X;

%mutiple each row by corresponding y values
for v=1:m
    A1(v,:)= A1(v,:)*(Y(v));
end
A1=A1*-1;

A(1:m, 1:n2)=1;
A(1:m, 1:n+1)=A1;
negI= eye(m)*-1;
A(1:m, n+2:n2)=negI;

d= ones(1,m)';
d= d*-1;

Aeq=[];
beq=[];

ub= [];
lb=zeros(1,n2)';
for h=1:n+1
    lb(h)=-Inf;
end


%comput w and b
[z,~,~,~,lamdas]=quadprog(H,f,A,d,Aeq,beq,lb,ub,[],opts);

index=0;
for t=1:m
    if(lamdas.ineqlin(t)>0)
        index=index+1;
        svs(index)=t;
    end
end

%set w and b values
b=z(1);

w= (1:n);
for i=2:n+1
    w(i-1)= z(i);  
end

als= (1:m);
for r=n+2:n2
    als(r-n-1)= z(r);  
end

w=w';
svs=svs';

end