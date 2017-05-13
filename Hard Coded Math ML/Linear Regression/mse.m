function err = mse(pred_Y, correct_Y)
% This function calculates the Mean Squared Error given two sets of output
% values, one set corresponding to the correct values, the other set
% representing the output values predicted by a regression model
% INPUT:
%  pred_Y: [m x 1] vector, containing the predicted values
%  correct_Y: [m x 1] vector, containing the correct values
%
% OUTPUT:
%  err: Mean Squared Error (scalar value)
%
[pM, ~]= size(pred_Y);

err=(pred_Y - correct_Y)'*(pred_Y - correct_Y)/pM;

end