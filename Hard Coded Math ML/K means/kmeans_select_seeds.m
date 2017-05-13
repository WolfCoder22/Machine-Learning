function seeds_idx = kmeans_select_seeds(X, K, mode)
% Returns an initial set of centroids (i.e. a set of seeds) for the Kmeans algorithm. 
%
% INPUT:
%  X: [m x n] matrix, where each row is a d-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of centroids (i.e. hyperparameter "K" in K-means)
%  mode: string, indicating the type of initilization. It can be either 'random' or 'diverse_set'.
% 
% OUTPUT:
%  seeds_idx: [1 x K] vector, containing the indices of the examples that 
%                     will be used as initial centroids; seeds_idx(i)
%                     should be an integer number between 1 and m.

X = X';
[~, m] = size(X);
if strcmp(mode, 'random')
    % random initialization
    seeds_idx = randperm(m);
    seeds_idx = seeds_idx(1:K);
elseif strcmp(mode, 'diverse_set')
    % WRITE YOUR CODE HERE
    X=X';
    seeds_idx(1)= 1;
    
    
    %loop though all k values
    for kCurr=2:K
        a=X;
        b=X(seeds_idx,:);
        
        [dist, ~]=min(kmeans_dist2(X, X(seeds_idx,:)), [], 2);
        [~, I]= sort(dist, 'descend');
        seeds_idx(kCurr)=I(1);
    end
else
  error('parameter mode not recognized');
end

end