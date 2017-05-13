function d = svm_kernel(x1, x2, mode)
% Calculates the kernel distance between two given examples.
% 
% INPUT:
%  x1: [1 x n] vector which is a n-dimensional example
%  x2: [1 x n] vector which is a n-dimensional example
%  mode: the type of kernel; it is a string that can be either 'linear' or 'polynomial'
%        Please see the homework text for further details.
%        
% OUTPUT:
%  d: [1 x 1] scalar value containing the kernel distance (either linear of polynomial)
%             between the two examples.

%computer linear kernel
if(strcmp(mode, 'linear'))

    d= x1*x2';
    
%comput polyno,ial kernel
else 
    d= (x1*x2')^3;
end


end