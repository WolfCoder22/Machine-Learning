function logprob = EM_logprobgauss(x, mu, sigma)
% Calculates the log-probability density value for the input example under a given multivariate Gaussian,
% i.e. log(P(x ; mu, sigma))
% 
% INPUT:
%  X: [1 x n] vector, representing an input example
%  mu: [n x 1] vector representing the mean of a Gaussian
%  sigma: [n x n] covariance matrix for the Gaussian
%
% OUTPUT:
%  l: [1 x 1] scalar value representing the log of the probability density value

[n, ~]= size(mu);
meanDiff=(x'-mu)';
sigInv=inv(sigma);
fromExp=-1/2*(meanDiff*sigInv*meanDiff');

determ= det(sigma);
siglog=(-1/2)*log(determ);

pilog=  (-n/2)*log(2*pi);

logprob=  fromExp+siglog+pilog;

end