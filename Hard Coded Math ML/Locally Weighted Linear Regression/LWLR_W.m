function W = LWLR_W(X, xtest, tau)
% Constructs the matrix W for solvigf weights later in train
%
% INPUT
% X     : [m x d] matrix, containing the d-dimensional input vectors 
%         of the m training examples
% xtest : [d x 1] vector, the input vector of a *single* test example
% tau   : scalar, a *single* value for the regularization hyperparameter

% OUTPUT
% W: [m x m] matrix

[m,~]= size(X);
m2=m;

WV= (1:m);
W=eye(m);

%loop through all x input vectors
for r=1:m
    
    %get the norm of the x difference
    xRow=X(r,:)';
    
    preNorm=xtest-xRow;
    N= norm(preNorm);

    WV(r)= exp(-1*(N^2/(2*tau^2)))/2;
    
end

%normalize the vectors in W
total=sum(WV);

WV=WV/total;

%create diagonal matrix W
for r=1:m2
    
    W(r,r)= WV(r);
end

end