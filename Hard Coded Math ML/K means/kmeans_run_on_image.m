function kmeans_run_on_image()
% This script requires the following functions to be implemented:
% kmeans_dist2
% kmeans_select_seeds
% kmeans
% kmeans_reconstructimgfromVQ

% K parameters for the Kmeans
Kvalues = [2 4 8];

% read and visualize the image
I = double(rgb2gray(imread('dartmouthhall.jpg')));
figure(2);
subplot(1,numel(Kvalues)+1,1);
imshow(uint8(I));
title('original image');

% split the image into tiles
tilesize = 8;
[X, num_x_tiles, num_y_tiles] =  kmeans_splitimgintiles(I, tilesize);

% run K-means for different numbers of centroids
for j=1:length(Kvalues),
    % execute Kmeans
    init_mode = 'diverse_set';
    seeds_idx = kmeans_select_seeds(X, Kvalues(j), init_mode);
    [tileidx, prototypes, distortions] = kmeans(X, Kvalues(j), seeds_idx);
    
    % reconstruct the image from its VQ form, and calculate the SSD.
    recI{j} = kmeans_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles);
    ssd = sum((I(:)-recI{j}(:)).^2);
    
    % visualize the reconstruction
    figure(2);
    subplot(1,numel(Kvalues)+1,j+1);
    imshow(uint8(recI{j}));
    title(['K = ' num2str(Kvalues(j)) '\newline ssd = ' sprintf('%e',ssd)]);
    fprintf('init_mode=%s; SSD using K=%d: %e\n', init_mode, Kvalues(j), ssd);
end

% save the plot (Note: do not remove this line of code)
saveas(gcf, 'q5a.fig');

end