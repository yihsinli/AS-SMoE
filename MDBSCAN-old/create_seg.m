E = 20;
data_file = 'pristine_images/'
%data_file = 'data/'
files = dir([data_file,'label',int2str(E),'/*.mat']);
%%
tic;
for i = 1:length(files)
    c=1;
    name=files(i).name;
    label = load([data_file,name]);
    im = imread([data_file,name]);
    %im = imgaussfilt(im,0.2);
    img = uint8(im);
    for E = 20:10:30
    %n = 300;
        
        
        
        if ~isfile([data_file,'label',int2str(E)])
            mkdir([data_file,'label',int2str(E)])
        end
        save([data_file,'label',int2str(E),'/',name(1:end-4),'.mat'],'label');
        imwrite(drawregionboundaries(label,img,[0,255,0]),[data_file,'label',int2str(E),'/',name(1:end-4),'.jpg'],'jpg')
        c=c+1;
    end
end
T = toc;
T/length(files)/length(20:10:100)