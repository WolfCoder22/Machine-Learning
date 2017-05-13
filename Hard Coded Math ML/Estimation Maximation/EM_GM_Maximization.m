function [mus, sigmas, priors, free_energy_m, likelihood_m] = EM_GM_Maximization(X, prob_c)
% Executes Maximization-step for the learning of a GMM.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  prob_c: [K x m] matrix, containing the the posterior probabilities over the K Gaussians for the m examples.
%                  Please see the comments in q5_GM_Expectation.m
%
% OUTPUT:
%  mus: [n x K] matrix containing the n-dimensional means of the K gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian.
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%  free_energy_m: [1 x 1] scalar value representing the free energy value
%  likelihood_m: [1 x 1] scalar value representing the log-likelihood value

[m,n]= size(X);
[K,~]=size(prob_c);
mus(1:n, 1:K)=0;
sigmas(1:n, 1:n, 1:K)=0;

priors=nansum(prob_c, 2)';

%for each gaussian find values
for i=1:K
    muTop=0;
    for v=1:m
        muTop= muTop + (X(v,:)'*prob_c(i,v));
    end
    
    mus(:,i)=muTop/priors(i);
    zigma(1:n, 1:n)=0;
    
    %get covarance matrix
    for j=1:m
        diff=X(j,:)'-mus(:,i);
        toAdd=prob_c(i, j)*(diff*diff');
        zigma=zigma+toAdd;
    end
    
    sigmas(:,:,i)=zigma/priors(i);  
end

%get the free energy value and loglik
fe=0;
ll=0;
for h=1:m
    lIter=0;
    for c=1:K
        qi=prob_c(c, h);
        
        if(qi>eps)
        fe=fe+ qi*(EM_logprobgauss(X(h,:), mus(:,c), sigmas(:,:,c))+log(priors(c))-log(qi));
        lIter=lIter+priors(c)*exp(EM_logprobgauss(X(h,:), mus(:,c), sigmas(:,:,c)));
        end
    end
    ll=ll+log(lIter);
end

free_energy_m=fe;
likelihood_m=ll;
priors=priors./m;

end