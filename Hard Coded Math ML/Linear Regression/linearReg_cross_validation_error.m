function error = linearReg_cross_validation_error(X, Y, lamda, mode, N)
% Calculates the cross-validation errors for different values of lambda, given the
% training set X, Y.
%
% ** Implementation notes **
% - As discussed in class, you should first randomly permute the examples, before starting the
%   cross-validation stage. Here we do it for you: use the vector idxperm
%   to index the examples
% - In the cross-validation stage, the indexes of the examples for the k-th subset must be 
%   idxperm([floor(m / N * k + 1) : floor(m / N * (k + 1))])
%   where k \in {0, 1, ..., N-1}
% - Do not change/initialize/reset the Matlab pseudo-number generator.
%
% INPUT
%  X: [m x d] matrix, where each row is a d-dimensional data example
%  Y: [m x 1] vector, where the i-th element is the ground truth target value for the i-th example. 
%  lambda: [1 x K] vector, containing the set of regularization hyperparameter values
%  mode: type of features, either 'linear' or 'quadratic'
%  N: number of folds for the cross-validation stage
%
% OUTPUT
%  error: [1 x K] vector containing the cross-validation error (i.e., the average of the mean 
%         squared errors over the N validation sets) for each lambda.
%


% ********  DO NOT TOUCH THE FOLLOWING 3 LINES  ********************
rand('twister', 0);
[m,  d] = size(X);
idxperm = randperm(m);
% ******************************************************************

error=lamda;

%randomly permute examples
Xcopy=X;
Ycopy=Y;
for j=1:m
    X(j,:)=Xcopy(idxperm(j),:);
    Y(j,:)=Ycopy(idxperm(j),:);
end

%get number of lamda's
lamdaSize= size(lamda);
K=lamdaSize(2);

%loop through every lamda value and number of folds
for currLamda=1:K
    err=0;
    for k=0:N-1
        
        %get cross valdiation test index
        i1= floor(m / N * k + 1);
        i2= floor(m / N * (k + 1));
        
        %get correct values for test segments
        xTest=X(i1 : i2,:);
        yTest=Y(i1 : i2,:);
        
        %get training segments
        if (i1==1)
            xTrain=X(i2+1:m,:);
            yTrain=Y(i2+1:m,:);

        elseif (i1==1)
            xTrain=X(1:i1-1,:);
            yTrain=Y(1:i1-1,:);
        else
           xTrain=X([1: i1-1 i2+1:m],:);
           yTrain=Y([1: i1-1 i2+1:m],:);
        end
        
        %get the error for this subsegment
        theta = linearReg_train(xTrain, yTrain, lamda(currLamda), mode);
        pred_Y = linearReg_predict(theta, xTest, mode);
        err = err+mse(pred_Y, yTest);
    end
    
    %get average error and store value
    error(currLamda)=err/N;
end

end
