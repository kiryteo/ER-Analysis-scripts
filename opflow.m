
for i=1:31
for j=0:8

%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t00' int2str(j) '_ch00.tif'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t00' int2str(j+1) '_ch00.tif'];
n1 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t00' int2str(j) '.tif'];
n2 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t00' int2str(j+1) '.tif'];
im1 = imread(n1);
im1 = mat2gray(im1);
im2 = imread(n2);
im2 = mat2gray(im2);
fl = opticalFlow(im1, im2);
[xg,yg]=meshgrid(1:128,1:128);
c=curl(xg,yg,fl.Vx,fl.Vy);
d=divergence(xg,yg,fl.Vx,fl.Vy);
%m = fl.Magnitude;
fname = ['/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/intensity_based_flow_curl/Ct' int2str(i) '_' int2str(j) '_opfl_curl.mat'];
save(fname, 'c');
fnamed = ['/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/intensity_based_flow_div/Ct' int2str(i) '_' int2str(j) '_opfl_div.mat'];
save(fnamed, 'd');
end;
end;



%imagesc(c);

%d=divergence(xg,yg,fl.Vx,fl.Vy);


for i=1:31
for j=9:9
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t00' int2str(j) '_ch00.tif'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t0' int2str(j+1) '_ch00.tif'];
n1 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t00' int2str(j) '.tif'];
n2 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t0' int2str(j+1) '.tif'];
im1 = imread(n1);
im1 = mat2gray(im1);
im2 = imread(n2);
im2 = mat2gray(im2);
fl = opticalFlow(im1, im2);
[xg,yg]=meshgrid(1:128,1:128);
c=curl(xg,yg,fl.Vx,fl.Vy);
d=divergence(xg,yg,fl.Vx,fl.Vy);
%m = fl.Magnitude;
fname = ['/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/intensity_based_flow_curl/Ct' int2str(i) '_' int2str(j) '_opfl_curl.mat'];
save(fname, 'c');
fnamed = ['/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/intensity_based_flow_div/Ct' int2str(i) '_' int2str(j) '_opfl_div.mat'];
save(fnamed, 'd');
end;
end;


for i=1:31
for j=10:98
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t0' int2str(j) '_ch00.tif'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t0' int2str(j+1) '_ch00.tif'];
n1 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t0' int2str(j) '.tif'];
n2 = ['/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_' int2str(i) '_decon_t0' int2str(j+1) '.tif'];
im1 = imread(n1);
im1 = mat2gray(im1);
im2 = imread(n2);
im2 = mat2gray(im2);
fl = opticalFlow(im1, im2);
[xg,yg]=meshgrid(1:128,1:128);
c=curl(xg,yg,fl.Vx,fl.Vy);
d=divergence(xg,yg,fl.Vx,fl.Vy);
%m = fl.Magnitude;
fname = ['/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/intensity_based_flow_curl/Ct' int2str(i) '_' int2str(j) '_opfl_curl.mat'];
save(fname, 'c');
fnamed = ['/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/intensity_based_flow_div/Ct' int2str(i) '_' int2str(j) '_opfl_div.mat'];
save(fnamed, 'd');
end;
end;


%for j=0:8
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t00' int2str(j) '_ch00_junc.png'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t00' int2str(j+1) '_ch00_junc.png'];
%im1 = imread(n1);
%im1 = mat2gray(im1);
%im2 = imread(n2);
%im2 = mat2gray(im2);
%fl = opticalFlow(im1, im2);
%m = fl.Magnitude;
%fname = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junc_opflow/R1_' int2str(j) '_opfl.mat'];
%save(fname, 'm');
%end;


%for j=9:9
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t00' int2str(j) '_ch00_junc.png'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t0' int2str(j+1) '_ch00_junc.png'];
%im1 = imread(n1);
%im1 = mat2gray(im1);
%im2 = imread(n2);
%im2 = mat2gray(im2);
%fl = opticalFlow(im1, im2);
%m = fl.Magnitude;
%fname = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junc_opflow/R1_' int2str(j) '_opfl.mat'];
%save(fname, 'm');
%end;

%for j=10:98
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t0' int2str(j) '_ch00_junc.png'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junctions/R1/R1_decon_t0' int2str(j+1) '_ch00_junc.png'];
%im1 = imread(n1);
%im1 = mat2gray(im1);
%im2 = imread(n2);
%im2 = mat2gray(im2);
%fl = opticalFlow(im1, im2);
%m = fl.Magnitude;
%fname = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junc_opflow/R1_' int2str(j) '_opfl.mat'];
%save(fname, 'm');
%end;


%for j=1:1
%for i=0:8
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A' int2str(j) '_decon_t009_ch00.tif'];
%n1 = ['/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A' int2str(j) '_decon_t00' int2str(i) '_ch00.tif'];
%im1 = imread(n1);
%im1 = mat2gray(im1);
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A' int2str(j) '_decon_t010_ch00.tif'];
%n2 = ['/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A' int2str(j) '_decon_t00' int2str(i+1) '_ch00.tif'];
%im2 = imread(n2);
%im2 = mat2gray(im2);
%fl = opticalFlow(im1, im2);
%plot(fl,'DecimationFactor',[2 2],'ScaleFactor',2)

%Vx = fl.Vx;
%y = fl.Vy;

%fnx = ['/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/flow_mag/A' int2str(j) '_' int2str(i) '_magX.mat'];
%fny = ['/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/flow_mag/A' int2str(j) '_' int2str(i) '_magY.mat'];
%save(fnx, 'Vx');
%save(fny, 'Vy');

%[xg,yg]=meshgrid(1:128,1:128);
%c=curl(xg,yg,fl.Vx,fl.Vy);

%imagesc(c);

%d=divergence(xg,yg,fl.Vx,fl.Vy);
%fname = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/opflow_div/R' int2str(j) '_9_div_flow.mat'];
%fname = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/opflow_div/R' int2str(j) '_' int2str(i) '_div_flow.mat'];
%save(fname, 'd');

%h = histogram(c(:));
%features(end+1) = c;
%ht = [ht, h];

%end;
%end;
