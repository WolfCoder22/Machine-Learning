function theta = LogisticReg_initialize(Xtrain, Ytrain, opt)
% Initializes the weights for training logistic regression

% INPUT
%  Xtrain  : [m x n] matrix, where each row is a n-dimensional input example (assume it 
%            already contains the constant feature set to 1)
%  Ytrain  : [m x 1] vector, where the i-th element is the correct label
%                    for the i-th example
%  opt     : string, can be either 'random' or 'heuristic' which allows to
%                    choose the initialization between randomly of heuristic

% OUTPUT
%  theta   : [n x 1] the initialized parameter vector

% HINTS
%  We provide the code for random initialization and ask you to implement
%  the case of 'heuristic’, which we have discussed in class.


n = size(Xtrain,2);
if strcmp(opt,'random')
    % random initialization
    rand('seed', 0);
    theta = rand(n,1); % generate initial value
else
    % "heuristic" initialization
    
    [m, ~]= size(Xtrain);
    Wtrain= ones(1,m)';
    thetaXproduct=ones(1,m);
    
    %change y values to .95 and .05 based of y
    for y=1:m
        if(Ytrain(y)==1)
            Wtrain(y)=.95;
        else
            Wtrain(y)=.05;
        end
    end
    
    %find what X*theta should be = from logistic equation
    for w=1:m
        thetaXproduct(w)= -1*(log((1/Wtrain(w))-1));
    end
    
    theta=thetaXproduct/Xtrain';
    theta=theta';
    
end



end
