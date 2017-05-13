function word_idx = nb_top_words(phi_y0, phi_y1, phi_prior, k)
% For each class, finds the words that are most indicative of a message
% belonging to that class

% INPUT
%  phi_y0    : [n x 1] vector, class conditional probabilities for y=0,
%              where phi_y0(j) = p(x_j = 1 | y = 0)
%  phi_y1    : [n x 1] vector, class conditional probabilities for y=1, 
%              where phi_y0(j) = p(x_j = 1 | y = 1)
%  phi_prior : [1 x 1] scalar, prior probability of y being 1, i.e., phi_prior = p(y = 1)
%  k         : [1 x 1] scalar, the number of words to output

% OUTPUT
%  word_idx  : [2 x k] matrix, the first row contains the indices of the k most indicative 
%              words for class y=0, the the second row the ones for y=1

[p,~]= size(phi_y0);
allProbs0= (1:p);
allProbs1= (1:p);
word_idx(1:2, 1:k)=0;

%loop through all features
for n=1:p
    
    top0= (1-phi_prior)* phi_y0(n);
    top1= (phi_prior)* phi_y1(n);
    bot= top0+top1;
    
    allProbs0(n)= top0/bot;
    allProbs1(n)= top1/bot;
end

%sort probs
[~, I0]=sort(allProbs0, 'descend');
[~, I1]=sort(allProbs1, 'descend');

word_idx(1,:)= I0(1:6);
word_idx(2,:)= I1(1:6);

end
