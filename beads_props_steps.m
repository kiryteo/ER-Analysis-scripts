% MATLAB Steps for beads properties

function props = beads_props_steps(filename)


img = imread(filename);
grimg = im2gray(img);
thr = multithresh(grimg, 2);
seg = imquantize(img, thr);
op = label2rgb(seg);
grop = im2gray(op);
beads = (grop == max(grop));
beads = double(beads);

bw = bwconncomp(beads, 8);
labeled = labelmatrix(bw);
props = regionprops('table', labeled, 'Area', 'Circularity', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'MajorAxisLength', 'MaxFeretProperties', 'MinFeretProperties', 'MinorAxisLength', 'Orientation', 'Perimeter');

%dlmwrite('/localhome/asa420/MIAL/Series005_decon_ch01.txt', props, 'delimiter', ' ');



writetable(props, '/localhome/asa420/MIAL/Live-STED_Series10_avg.txt');
% Further props are possible with passing grayscale image, image intensity specific props
