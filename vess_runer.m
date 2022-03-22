%for i=11:11
%dirname = ['/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/Series0' int2str(i) '_decon_converted/erode/'];
%for j=13:99
%fname = [dirname 'Series0' int2str(i) '_decon_converted_t' int2str(j) '_ch00_std_erode.png'];
%Vessel2d(fname);
%end;
%end;


for i=1:1
fname = ['/localhome/asa420/ER-Analysis-scripts/erode-mean-proj/C' int2str(i) '_er.png'];
Vessel2d(fname);
end;
