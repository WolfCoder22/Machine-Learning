function error = LogisticReg_error(Y, pred_Y)
% Calculates the misclassification rate by comparing the predicted labels pred_Y to
% the true labels Y

% INPUT
%  Y     : [m x 1] vector, ground truth labels
%  pred_Y: [m x 1] vector, predicted labels

% OUTPUT
%  error : [1 x 1] scalar, misclassification rate, i.e. the number of
%  examples misclassified over the total number of examples
    
[m,~]= size(Y);

correctNum=0;

for i=1:m
    if(Y(i)==pred_Y(i))
        correctNum= correctNum+1;
    end
end

error= 1-correctNum/m;

end
