function recI = kmeans_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles)
% Reconstructs an image starting from the VQ model.
%
% INPUT:
%  prototypes: [n x K] matrix, containing the n-dimensional centroids of the K clusters.
%  tilesize: [1 x 1] scalar, indicating the size of the tiles.
%  tileidx: [m x 1] vector, containing the labels that the Kmeans algorithm assigned to the data.
%           tileidx(i) is an element of {1 ... K} and it indicates the
%           cluster/prototype ID associated to the i-th example/tile;
%           not that tileidx stores the tiles in column order (see comments in file q5_splitimgintiles.m)
%  num_x_tiles: [1 x 1] scalar value, indicating the number of tiles along the x axis.
%  num_y_tiles: [1 x 1] scalar value, indicating the number of tiles along the y axis.
% 
% OUTPUT:
%  recI: [r x c] matrix, corresponding to the reconstructed gray-scale image

r=tilesize*num_x_tiles;
c=tilesize*num_y_tiles;

recI(1:r, 1:c)=0;

tileNumber=0;

startX=1;
endX=tilesize;


%loop tile rows
for x=1:num_x_tiles;
    
    startY=1;
    endY=tilesize;

    %loop tile collums
    for y=1:num_y_tiles
        
        tileNumber=tileNumber+1;
        
        tileVector= prototypes(:,tileidx(tileNumber));
        tile=eye(tilesize);
        
        start=1;
        endI=tilesize;
        
        %create WxW tile from vector
        for i=1:tilesize
            tile(:,i)= tileVector(start:endI);
            start=start+tilesize;
            endI=endI+tilesize;
        end
        
        %update output image
        recI(startY:endY, startX:endX)= tile;
        
        startY=startY+tilesize;
        endY=endY+tilesize;

        
    end
    
    startX=startX+tilesize;
    endX=endX+tilesize;
end
    
end