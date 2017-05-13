function [phi_y0, phi_y1, phi_prior] = nb_train(Xtrain, Ytrain)
% Train a Naive Bayes model using Laplacian smoothing

% INPUT
%  Xtrain    : [m x n] matrix, where each row is an n-dimensional input *training* example
%  Ytrain    : [m x 1] vector, where the i-th element is the label for the i-th *training* example

% OUTPUT
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)

[m,n]= size(Xtrain);
phi_y0= (n:1);
phi_y1= (n:1);
phi_prior=0;

%loop through all examples
for i=1:m
    if(Ytrain(i)==1)
        phi_prior= phi_prior+1;
    end
end

phi_prior= phi_prior/m;

%go through all features
for f=1:n
    phiN00=0;
    phiN01=0;
    phiN10=0;
    phiN11=0;
    
    %loop through all examples
    for w=1:m
        
        %get number of {X=1^y=1}
        if(Xtrain(w,f)==1 && Ytrain(w)==1)
            phiN11=phiN11+1;
        %get number of {X=0^y=1}
        elseif(Xtrain(w,f)==0 && Ytrain(w)==1)
            phiN01=phiN01+1;
        %get number of {X=1^y=0}
        elseif(Xtrain(w,f)==1 && Ytrain(w)==0)
            phiN10=phiN10+1;
        %get number of {X=0^y=0}
        else
            phiN00=phiN00+1;
        end
    end
    
    %find phi values with lapacian smoothing
    phi_y1(f)= (1+phiN11)/(1+phiN11+ 1+phiN01);
    phi_y0(f)= (1+phiN10)/(1+phiN10+ 1+phiN00);

end

phi_y1=phi_y1';
phi_y0=phi_y0';

end
