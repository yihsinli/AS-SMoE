%======================================================================
% DBSCAN demo in TIP 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % J. Shen, X. Hao, Z. Liang, Y. Liu, W. Wang, and L. Shao, 
% % Real-time Superpixel Segmentation by DBSCAN Clustering Algorithm, 
% % IEEE Trans. on Image Processing, 25(12):5933-5942, 2016 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2016  Beijing Laboratory of Intelligent Information Technology
% Any further questions, please send email to: shenjianbing@bit.edu.cn  or shenjianbingcg@gmail.com
%======================================================================
%Input parameters are: 
%[1] 8 bit images (color)
%[2] Number of required superpixels (optional, default is 200)
%[3] post processing(1 is need,0 is not)
%Ouputs are:
%[1] labels (in raster scan order)

%NOTES:
%[1] number of returned superpixels may be different from the input
%number of superpixels.
%[2] you should compile the cpp file using visual studio 2008 or later version
% -- mex DBscan_mex.cpp 
%======================================================================
% clear all;
% close all;
% names = ["train_001","train_002","train_003","train_004"]
% currentFolder = pwd;
% addpath(genpath(currentFolder))
% %addpath('code');
% addpath('../data/');
clear all;

post=1;
files = dir('data/*.jpg');
n = 150;
for i = 1:length(files)
    name=files(i).name;
    im = imread(['data/',name]);
    img = uint8(im);
    number_superpixels = n;
    for n = 100:50:500
        tic;
        
        if ~isfile(['./label/',int2str(n)])
            mkdir(['./label/',int2str(n)])
        end
        label = DBscan_mex(img,n,post);
        save(['./label/',int2str(n),'/',name(1:end-4),'.mat'],'label');
        toc;
        %SuperpixelSave(label,im,name);
        %DisplaySuperpixel(label,im,['./label/150/',sprintf('%s.jpg',name)]);
    end
end
% end
