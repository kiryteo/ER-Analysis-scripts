for i=1:1
%path = '/localhome/asa420/MIAL/aggregation-with-median/';
%name = [path 'RTN/max/files/RTNSeries' int2str(i) '-max.png'];
%bname = [path 'RTN/RTN-brpts' int2str(i) '-max.png'];
%thname = [path 'RTN/RTN-brpts' int2str(i) '-avg-th-norm.png'];
for j=0:0
filename = ['/localhome/asa420/Desktop/ATL/std/A' int2str(i) '_decon_t00' int2str(j) '_ch00_std.png'];
mchname = ['/localhome/asa420/Desktop/ATL/mcherry/A' int2str(i) '/A' int2str(i) '_decon_t00' int2str(j) '_ch01_std_adj.png'];
%skname = ['/localhome/asa420/Desktop/ATL/skel/A' int2str(i) '/A' int2str(i) '_decon_t00' int2str(j) '_ch00_skel.png'];
%brname = ['/localhome/asa420/Desktop/ATL/brpts/A' int2str(i) '/A' int2str(i) '_decon_t00' int2str(j) '_ch00_skel_brpts.png'];
sample_creator(filename, mchname);
end
end
