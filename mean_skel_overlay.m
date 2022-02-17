
function op = mean_skel_overlay(inputdir)
s = dir(fullfile(inputdir, '*.tif'));

for i=1:numel(s)
  filename = fullfile(inputdir, s(i).name);
  frame = imread(filename);
  skel = imread('/localhome/asa420/MIAL/aggregation-with-median/RTN/avg/skel/RTNSeries16-avg_skel.png');
  op = imfuse(frame, skel);
  opname = [fullfile(inputdir, s(i).name) '_mean_skel.tif'];
  mon = montage({frame, skel, op}, 'size', [1 3], 'BorderSize', [1 1], 'BackgroundColor', 'white');
  mon_im = mon.CData;
  imwrite(mon_im, opname);
end
