function SuperpixelSave(label,img,name)   
imwrite(img,[name,'result.bmp'],'bmp')
%   imwrite(img,['BSDS300_result\result\DBSCAN\result1\',name,'.bmp'],'bmp')
    
fid=fopen([name,'.txt'],'wt');%д���ļ�·��
[m,n]=size(label);
 for i=1:1:m
    for j=1:1:n
       if j==n
         fprintf(fid,'%g\n',label(i,j));
      else
        fprintf(fid,'%g\t',label(i,j));
       end
    end
end
fclose(fid);

end

