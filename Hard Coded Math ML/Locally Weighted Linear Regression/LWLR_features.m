function B = linearReg_features(X, mode)
% Given the data matrix X (where each row X(i,:) is an example), the function
% computes the feature matrix B, where row B(i,:) represents the feature vector 
% associated to example X(i,:). The features should be either linear or quadratic
% functions of the inputs, depending on the value of the input argument 'mode'.
% Please make sure to implement the features according to the *exact* order
% specified in the text of the homework assignment.
%
% INPUT:
%  X: a matrix [m x d] where each row is a d-dimensional input example
%  mode: the type of features; it is a string that can be either 'linear' or 'quadratic'.
%
% OUTPUT:
%  B: a matrix [m x n], with each row containing the feature vector of an example
%

if ~strcmp(mode, 'linear') && ~strcmp(mode, 'quadratic')
  disp('Error, only linear and quadratic features are supported');
end


%linear computation of feature vectors
if strcmp(mode, 'linear')
  
  dimen= size(X);
  height=dimen(1);
  width=dimen(2)+1;
  
  %add ones digit to B
  for r=1:height
      B(r,1)= 1;
  end
  
  %loop through rest of data
  for i=2:width
      for j=1:height
     
      B(j,i)= X(j, i-1);
      end
  end
  
%quadratic computation of feature vectors
else
    dimensions= size(X);
   
    xH= dimensions(1);
    xW= dimensions(2);
    
    newXW=xW+1;
    for w=1:xW
        newXW=newXW+w;
    end
    
    %initialize Matrix
    B(1:xH, 1:newXW)=1;
 
    
    %loop through current data and add to B
    for i=2:xW+1
       for j=1:xH

        B(j,i)= X(j, i-1);
        end
    end
    
    collNum= xW+2;

    %loop every row
    for r=1:xH
        
        %loop through every element in every row
         for x1=2:xW+1
             elem1= B(r,x1);
             
             for x2=x1:xW+1
                 elem2= B(r,x2);

                 productX= elem2*elem1;
                 
                 %assign quadratic value to feature vector
                 B(r,collNum)= productX;
                 
                 collNum=collNum+1;
             end
         end
         
         collNum=xW+2;
    end

     
end