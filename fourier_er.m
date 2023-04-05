a=zeros(128,128,100);

/localhome/asa420/ER-Analysis-data/A1_er.mp4




A=fftn(a-mean(a(:)));
Aabs= abs(A);
Aviz=fftshift((mat2gray(abs(A))));
AvizTime = squeeze(sum(sum(Aviz,2),1));
xlabel('Frequency');ylabel('|Fourier(f)|')




%histogram(log(mat2gray(Aabs(:))))
%montage(Aabs)
%Aviz=fftshift(log(mat2gray(abs(A))));

Aviz=fftshift((mat2gray(abs(A))));
AvizTime = squeeze(sum(sum(Aviz,2),1));



plot(-50:49,AvizTime,'LineWidth',3);
xlabel('Frequency');ylabel('|Fourier(f)|')
%isosurface(Aviz,0.5)
if 0
    montage(Aviz,'BorderSize',[3,3])
end
