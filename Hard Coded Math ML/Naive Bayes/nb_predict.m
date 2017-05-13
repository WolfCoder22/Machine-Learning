function pred_Y = nb_predict(X, phi_y0, phi_y1, phi_prior)
% Predicts the labels of examples X, given the trained model

% INPUT
%  X         : [m x n] matrix containing m examples, each corresponding to an n-dimensional row of the matrix
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)

% OUTPUT
%  pred_Y    : [m x 1] vector, predicted labels for the m examples in X

% HINTS
%  1. for an example pred_y = argmax_k p(y=k) \prod_{j=1}^n p(x_j|y=k) 
%  2. use the log function to avoid numerical problems:
%       pred_y = argmax_k { \log{p(y=k)} + \sum_{j=1}^n \log{p(x_j|y=k)} }

[m, n]= size(X);
pred_Y=(m:1);

%loop through all input vectors
for i=1:m
    
    py0= log(1-phi_prior);
    py1= log(phi_prior);
    
    %loop through all features
    for j=1:n
        
        %check x feature value=1
        if(X(i,j)==1)
            py0= py0+log(phi_y0(j));
            py1= py1+log(phi_y1(j));
            
        %x value was 0
        else
            py0= py0+log(1-phi_y0(j));
            py1= py1+log(1-phi_y1(j));
        end  
    end
    
    %predict classification based off arg max y
    if(py0>= py1)
        pred_Y(i)= 0;
    else
        pred_Y(i)= 1;
    end
end
pred_Y= pred_Y';
end
