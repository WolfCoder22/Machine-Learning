function [mus, sigmas, priors] = EM_gmminit(X, K, labels)
% Initializes a GMM model, given an initial clustering.

% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of gaussians for the GMM
%  labels: [m x 1] vector, containing the labels that the Kmeans algorithm assigned to the data.
%                  Each label l is an element of {1 ... K}, and it is associated with 
%                  the l-th gaussian.
% 
% OUTPUT:
%  mus: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                      covariance matrix for the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.

[m, n]= size(X);
priors= (1:K)'*0;
mus(1:n, 1:K)=0;
sigmas(1:n, 1:n, 1:K)=0;

%loop through all examples
for i=1:m
    
    l=labels(i);
    
    %update priors
    priors(l)=priors(l)+1;
    
    %update mus
    mus(:,l)=mus(:,l)+ X(i,:)';
    
end

%make averages for each K 
for j=1:K
    mus(:,j)= mus(:,j)/priors(j);
end

%compute covar 3d matrixs
for v=1:K
    matrix(1:n, 1:n)=0;
    for g=1:m
        l=labels(g);
        if(l==v)
            diff=X(g,:)'-mus(:,v);
            toAdd= diff*diff';
            matrix=matrix+toAdd;
        end
    end
    sigmas(:,:,v)=matrix/priors(v);
end

priors=priors/m;
priors=priors';

end

