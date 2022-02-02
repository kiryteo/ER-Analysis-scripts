% MATLAB Steps for beads properties

function properties = beads_props_steps(filename)

img = load(filename);

%grimg_conf = im2gray(img.npy(1,:,:));
%grimg_sted = im2gray(img.npy(2,:,:));
%grimg_synth = im2gray(img.npy(3,:,:));

grimg_conf = double(img.npy(1,:,:));
grimg_sted = double(img.npy(2,:,:));
grimg_synth = double(img.npy(3,:,:));

grimg_conf = reshape(grimg_conf, 256, 256);
grimg_sted = reshape(grimg_sted, 256, 256);
grimg_synth = reshape(grimg_synth, 256, 256);

thr_conf = multithresh(grimg_conf, 2);
thr_sted = multithresh(grimg_sted, 2);
thr_synth = multithresh(grimg_synth, 2);

seg_conf = imquantize(grimg_conf, thr_conf);
seg_sted = imquantize(grimg_sted, thr_sted);
seg_synth = imquantize(grimg_synth, thr_synth);

op_conf = label2rgb(seg_conf);
op_sted = label2rgb(seg_sted);
op_synth = label2rgb(seg_synth);

overlap1 = imfuse(op_sted, op_synth);
%imshow(overlap1);

%figure()

grop_conf = im2gray(op_conf);
grop_sted = im2gray(op_sted);
grop_synth = im2gray(op_synth);

%montage({grop_sted, grop_synth, overlap1}, 'size', [1 3]);

beads_conf = (grop_conf == max(grop_conf));
beads_sted = (grop_sted == max(grop_sted));
beads_synth = (grop_synth == max(grop_synth));

beads_conf = double(beads_conf);
beads_sted = double(beads_sted);
beads_synth = double(beads_synth);

%figure()
overlap2 = imfuse(beads_sted, beads_synth);
%imshow(overlap2);

%figure()
%montage({beads_sted, beads_synth, overlap2}, 'size', [1 3]);

bw_conf = bwconncomp(beads_conf, 8);
bw_sted = bwconncomp(beads_sted, 8);
bw_synth = bwconncomp(beads_synth, 8);

labeled_conf = labelmatrix(bw_conf);
labeled_sted = labelmatrix(bw_sted);
labeled_synth = labelmatrix(bw_synth);

conf_beads_num = max(labeled_conf(:));
sted_beads_num = max(labeled_sted(:));
synth_beads_num = max(labeled_synth(:));



%props_conf = regionprops('table', labeled_conf, 'Area', 'Circularity', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'MajorAxisLength', 'MaxFeretProperties', 'MinFeretProperties', 'MinorAxisLength', 'Orientation', 'Perimeter');
%props_sted = regionprops('table', labeled_sted, 'Area', 'Circularity', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'MajorAxisLength', 'MaxFeretProperties', 'MinFeretProperties', 'MinorAxisLength', 'Orientation', 'Perimeter');
%props_synth = regionprops('table', labeled_synth, 'Area', 'Circularity', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'MajorAxisLength', 'MaxFeretProperties', 'MinFeretProperties', 'MinorAxisLength', 'Orientation', 'Perimeter');

%series_name = split(filename, '/');
%fname_ext = series_name(end);
%fname = split(fname_ext, '.');
%new_conf = strcat(fname(1), '-conf', '.csv');
%new_sted = strcat(fname(1), '-sted', '.csv');
%new_synth = strcat(fname(1), '-synth', '.csv');


%prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn-features/';

%name_conf = strcat(prefix, new_conf{1});
%name_sted = strcat(prefix, new_sted{1});
%name_synth = strcat(prefix, new_synth{1});

%writetable(props_conf, name_conf);
%writetable(props_sted, name_sted);
%writetable(props_synth, name_synth);



% Further props are possible with passing grayscale image, image intensity specific props

% Running loop for directories
%rtndir = 'path/to/rtn/dir';
%files = dir(fullfile(rtndir));

%for k = 3:length(files)
%    basefile = files(k).name;
%    fullfname = fullfile(rtndir, basefile);
%    beads_props_steps(fullfname);
%end
