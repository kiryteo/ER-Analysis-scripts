for i=7:16
path = '/localhome/asa420/MIAL/aggregation-with-median/';
name = [path 'RTN/max/files/RTNSeries' int2str(i) '-max.png'];
bname = [path 'RTN/RTN-brpts' int2str(i) '-max.png'];
thname = [path 'RTN/RTN-brpts' int2str(i) '-avg-th-norm.png'];
sample_creator(name, bname, thname);
end
