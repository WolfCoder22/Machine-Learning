function nb_spamFilter()
% This script requires the following functions to be implemented:
% nb_train
% nb_predict
% nb_top_words

S = load('spamdata.mat');
X = S.trainsetX;
Y = S.trainsetY;

clear S;

k = 6;

feature_names = textread('spambase_names.txt', '%s');

[phi_y0, phi_y1, phi_prior] = nb_train(X, Y);

word_idx = nb_top_words(phi_y0, phi_y1, phi_prior, k);

for i=1:size(word_idx,1)
    str = feature_names{word_idx(i,1)};
    for j=2:size(word_idx,2)
        str = [str ', ' feature_names{word_idx(i,j)}];
    end
    fprintf('Top %d words for class %d: %s.\n', k, i-1, str);
end