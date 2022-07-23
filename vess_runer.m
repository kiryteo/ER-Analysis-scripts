

%fname = '/localhome/asa420/ER-Analysis-scripts/A18_aop6-op_loc.png'
%Vessel2d(fname);

for i=1:29
dirname = ['/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/preproc/R' int2str(i) '/'];
for j=10:99
fname = [dirname 'R' int2str(i) '_decon_t0' int2str(j) '_ch00_proc.png'];
Vessel2d(fname);
end;
end;


%for i=16:16
%dirname = ['/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series0' int2str(i) '_decon_converted/matching/'];
%for j=0:9
%fname = [dirname 'Series0' int2str(i) '_decon_converted_t0' int2str(j) '_ch00_std.png'];
%Vessel2d(fname);
%end;
%end;



%for i=1:1
%fname = ['/localhome/asa420/ER-Analysis-scripts/erode-mean-proj/C' int2str(i) '_er.png'];
%Vessel2d(fname);
%end;
