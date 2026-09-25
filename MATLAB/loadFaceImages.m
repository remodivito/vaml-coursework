function [images, labels,dimensions] = loadFaceImages(filename,sampling)

if nargin<2
    sampling =1;
end

% this is a flag that allow you to activate/deactivate the data augmentation
% Data augmentation will increase the size of the dataset by created variations 
%(mirroring, flipping, displacements) of each given image. This aims to produce more
% training images and, therefore, improve performance
augmented=1;


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);


line1=fgetl(fp);
line2=fgetl(fp);

numberOfImages = fscanf(fp,'%d',1);

images=[];
labels =[];
dimensions = [];
for im = 1:numberOfImages
    label = fscanf(fp, '%d', 1);
    imfile = fscanf(fp, '%s', 1);
    
    if mod(im-1, sampling) == 0
        I = imread(imfile);
        if size(I, 3) > 1
            I = rgb2gray(I);
        end
        vector = reshape(I, 1, []);
        vector = double(vector); % / 255;

        images = [images; vector];
        labels = [labels; label];
        dimensions = [dimensions; size(I, 1), size(I, 2)];


    
    
    if augmented
        
        if label==1
            Itemp =fliplr(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(I,1)
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(I,-1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(I,[0 1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(I,[0 -1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(fliplr(I),1)
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(fliplr(I),-1);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(fliplr(I),[0 1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =circshift(fliplr(I),[0 -1]);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
        else
            Itemp =fliplr(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =flipud(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
            
            Itemp =flipud(fliplr(I));
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label]; 
            dimensions = [dimensions; size(Itemp, 1), size(Itemp, 2)];
        end

    end

    else
        continue;
    end
    
    
end

disp(labels)

fclose(fp);

end
