function [X, num_x_tiles, num_y_tiles] =  kmeans_splitimgintiles(I, tilesize)
% Split the input image into [tilesize x tilesize] tiles.
%
% INPUT:
%  I: [r x c] matrix, corresponding to a gray-scale image
%  tilesize: [1 x 1] scalar, indicating the size of the tiles.
%
% OUTPUT:
%  X: [m x tilesize^2] matrix where each row is the vectorized version of a square tile.
%                      The i-th row correponds to the i-th tile of the image. The tiles are
%                      stored in column order, i.e. top to bottom, left to right.
%                      Note that we vectorize each tile in a column-wise 
%                      fashion (Matlab standard vectorization).
%  num_x_tiles: [1 x 1] scalar value, indicating the number of tiles along the x axis.
%  num_y_tiles: [1 x 1] scalar value, indicating the number of tiles along the y axis.

% ******************************************************************
% ****************** DO NOT EDIT THIS FUNCTION *********************
% ******************************************************************

[r,c] = size(I);

num_x_tiles = floor(c/tilesize);
num_y_tiles = floor(r/tilesize);

X = zeros(tilesize^2, num_x_tiles*num_y_tiles);

count = 1;
for j=1:num_x_tiles,
    for i=1:num_y_tiles,    
        X(:, count) = reshape(I((i-1)*tilesize+1:i*tilesize, (j-1)*tilesize+1:j*tilesize), ...
                              tilesize^2, 1);
        count = count + 1;
    end
end
X = X';     

end
 
 