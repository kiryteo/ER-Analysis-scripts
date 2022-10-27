%% load/crop the image
%I=imread('/Users/hamarneh/Downloads/3_18_2021 Control COS7 Decon_Series002_decon_ch02.tif');
%I=imread('/localhome/asa420/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/Series007_decon_converted/files/Series007_decon_converted_t02_ch00.tif');
I = imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A24_decon_t000_ch00.tif');

% crop a small part?
% A=x3_18_2021ControlCOS7Decon_Series002_decon_ch02;
% [x,RECT]=imcrop(I);imagesc(x);
% RECT = [1310.51       1499.51        911.98        653.98];
% x=imcrop(x,RECT);imagesc(x);

% no cropping
x=double(mat2gray(I));

%% sheets and tubules
sheets = medfilt2(x,[50 50],'symmetric');
sheets=mat2gray(sheets);
y = medfilt2(x,[10 10],'symmetric');
y=mat2gray(y);
tubules=y.*(sheets<0.1);

%% visualization
clr1 = zeros(size(x,1),size(x,2),3);
clr1(:,:,1)=0.1*x+0.9*tubules;
clr1(:,:,2)=0;
clr1(:,:,3)=0;

clr2 = zeros(size(x,1),size(x,2),3);
clr2(:,:,1)=0;
clr2(:,:,2)=0.1*x+0.9*sheets;
clr2(:,:,3)=0;

clr3 = zeros(size(x,1),size(x,2),3);
clr3(:,:,1)=0.5*x+0.5*tubules;
clr3(:,:,2)=0.5*x+0.5*sheets;
clr3(:,:,3)=0.5*x;

montage({x,clr1,clr1>0.2,clr3,clr2,clr2>0.2})
