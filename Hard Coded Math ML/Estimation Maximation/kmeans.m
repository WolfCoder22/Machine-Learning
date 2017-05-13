function [labels, means, distortions] = kmeans(X, K, seeds_idx)
% Executes Kmeans clustering algorithm, using euclidean distances.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  K: [1 x 1] scalar value, indicating the number of centroids (i.e. hyperparameter "K" in K-means)
%  seeds_idx: [1 x K] vector, containing the indices of the examples that 
%                     will be used as initial centroids.
% 
% OUTPUT:
%  labels: [m x 1] vector, containing the labels that the K-means algorithm assigned to the examples.
%                  labels(i) is an element of {1 ... K}, and it indicates the cluster ID associated to the i-th example
%  means: [n x K] matrix, containing the n-dimensional centroids of the K clusters.
%  distortions: [1 x num_iterations] vector, each element containing the total distortion at a particular iteration, i.e.
%                                    the sum of the squared Euclidean distances between the examples
%                                    and their associated centroids.

[m,n]= size(X);
means(1:n, 1:K)=0;

newlabels= ones(m,1)*0;

num_iterations=0;
sse=10;

%store values of centroirds centers
OldC(1:n, 1:K)=0;
for v=1:K
    OldC(:,v)= X(seeds_idx(v),:);
end

%converge until error less than 10^-6
while(sse>10^(-6))
   
    num_iterations=num_iterations+1;
    
    %update point label
    for i=1:m
        
        minDist=Inf;
        for k=1:K
            
            %get distance
            point=X(i,:);
            centroid= OldC(:,k)';
            kDist=(point-centroid)*(point-centroid)';
            
            if(kDist<minDist)
                minDist=kDist;
                newlabels(i)=k;
            end
        end
    end
   
    %update cluster centers
    for c=1:K
        
        num=zeros(1,n);
        den=0;
        for j=1:m
            
            if(newlabels(j)==c)
                num=num+ X(j,:);
                den= den+1;
            end
        end 
        
        means(:,c)=num/den;
    end
    
    diff=means-OldC;
    sse= sumsqr(diff);
    distortions(num_iterations)=sse;
    OldC=means;
end

labels=newlabels;
end
