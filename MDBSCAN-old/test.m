
%%
% for i = 1:length(files)
%     c=1;
%     name=files(i).name;
%     mim = load([data_file,name]);
%     mimg = uint8(mim);
%     imwrite(rgbImage, 'my image.jpg');
%     im = imread([data_file,name]);
%     %im = imgaussfilt(im,0.2);
%     img = uint8(im);


%%
E2 = 10;
partition = 'kodim_noisy_0.05';
data_file = ['../data/img','/',partition,'/'];
%data_file = 'data/'
files = dir([data_file '*.png']);
%%
c=0;
for i = 1:length(files)
    tic;
    
    name=files(i).name;
    im = imread([data_file,name]);
    if length(size(im)) == 2
        im = cat(3, im, im, im);
    end
    im = imgaussfilt(im,1); % diff = 10: var = 3; diff = 20: var = 1
    img = uint8(im);
    grayim = rgb2gray(img);
    grayim = [grayim grayim grayim];
    grayim = reshape(grayim,size(grayim,1),size(grayim,2)/3,3);

    for E = 20:10:31
    %n = 300;
        
        label = DBscan_mex(img,100,1,E,E);
        
        if ~isfile(['../data/seg/mdbscan/',int2str(E),'/',partition])
            mkdir(['../data/seg/mdbscan/',int2str(E),'/',partition])
        end
        save(['../data/seg/mdbscan/',int2str(E),'/',partition,'/',name(1:end-4),'.mat'],'label');
        imwrite(drawregionboundaries(label,grayim,[255,255,0]),['../data/seg/mdbscan/',int2str(E),'/',partition,'/',name(1:end-4),'_gray.jpg'],'jpg')
        
        %label=load('./label/500/train_001.mat');
        %label = label.label;

        %
        % initialization
        %[mylabel, myAm] = mcleanupregions(label, 0);
        %Am = regionadjacency(label);
        %Np = length(Am);
        %regionsC  = zeros(Np,1);
        %for n = 1:Np
        %    regionsC(n) = n;
        %end
        %%[Sp,Am,l] = updateL(label,regionsC,L,A,B);
        %%nim = showseg(label,regionsC,L,A,B);
        %%subplot(2,length(300:10:400),c+length(300:10:400)),imshow(uint8(nim));
        %%subplot(length(files),length(20:10:100),c+length(20:10:100)*(i-1)),imshow(drawregionboundaries(label,img,[0,255,0]))
        %%title([int2str(E)]);
        
        
        %'%d / %d T: %f'
        
        %T/length(files)/length(20:10:100)
    end
    c=c+1;
    T = toc;
    [int2str(c) ' / ' int2str(length(files)) ' ' num2str(T)]
end
T = toc;
T/length(files)/length(20:10:100);