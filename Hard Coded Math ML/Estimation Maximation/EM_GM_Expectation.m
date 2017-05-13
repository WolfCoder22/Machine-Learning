function [prob_c, free_energy_e, likelihood_e] = EM_GM_Expectation(X, mus, sigmas, priors)
% Executes the Expectation-step for the learning of a GMM.
%
% INPUT:
%  X: [m x n] matrix, where each row is an n-dimensional input example
%  mus: [n x K] matrix containing the n-dimensional means of the K Gaussians
%  sigmas: [n x n x K] 3-dimensional matrix, where each matrix sigmas(:,:,i) is the [n x n] 
%                           covariance matrix of the i-th Gaussian
%  priors: [1 x K] vector, containing the mixture priors of the K Gaussians.
%
% OUTPUT:
%  prob_c: [K x m] matrix, containing the posterior probabilities over the K Gaussians for the m examples.
%          Specifically, prob_c(j, i) represents the probability that the
%          i-th example belongs to the j-th Gaussian, 
%          i.e., P(z^(i) = j | X^(i,:))
%  free_energy_e: [1 x 1] scalar value representing the free energy value
%  likelihood_e: [1 x 1] scalar value representing the log-likelihood value

[m, ~]= size(X);
[~, K]= size(mus);
prob_c(1:K, 1:m)=0;

sum(1:m)=0;
%get prob_c
for u=1:m
    for r=1:K
        prob=exp(EM_logprobgauss(X(u,:), mus(:,r), sigmas(:,:,r)));
        prob_c(r,u)=prob*priors(r);
        sum(u)=sum(u)+prob_c(r,u);
    end
end

for f=1:K
    for g=1:m
    prob_c(f,g)= prob_c(f,g)/sum(g);
    end
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

free_energy_e=fe;
likelihood_e=ll;

end