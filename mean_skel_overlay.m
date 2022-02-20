
function op = mean_skel_overlay(inputdir)
s = dir(fullfile(inputdir, '*.tif'));

for i=1:numel(s)
  filename = fullfile(inputdir, s(i).name);
  frame = imread(filename);
  rgb_frame = cat(3, frame, frame, frame);
  rgb_frame(:,:,1) = 0;
  rgb_frame(:,:,3) = 0;
%  size(rgb_frame)
  avgframe = imread('/localhome/asa420/MIAL/aggregation-with-median/RTN/avg/files/RTNSeries16-avg.png');
  rgb_avg = cat(3, avgframe, avgframe, avgframe);
  rgb_avg(:,:,2) = 0;
  skel = imread('/localhome/asa420/MIAL/aggregation-with-median/RTN/avg/skel/RTNSeries16-avg_skel.png');
  rgb_skel = cat(3, skel, skel, skel);
  rgb_skel(:,:,2) = 0;
  op1 = imfuse(rgb_frame, rgb_avg);
  op2 = imfuse(rgb_frame, rgb_skel);
  opname = [fullfile(inputdir, s(i).name) '_mean_skel.tif'];
  mon = montage({rgb_frame, rgb_avg, op1, rgb_skel, op2}, 'size', [1 5], 'BorderSize', [1 1], 'BackgroundColor', 'white');
  mon_im = mon.CData;
  imwrite(mon_im, opname);
end
