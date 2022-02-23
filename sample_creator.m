%function op = create_samples(filename, brname, thname)
function op = create_samples(skname, juncname)

%[Rootname, Path, ~] = ExtractRootName(filename);
[Rootname, Path, ~] = ExtractRootName(skname);

%img = imread(filename);


%grimg = im2gray(img);
%bead_fname = [Path Rootname '_beads.png'];
%brpts_fname = [Path Rootname '_brpts.png'];
%brpts_fname = [Path 'Ctrl-brpts2-max.png'];
%brpts_fname = '/localhome/asa420/MIAL/aggregation-with-median/Control/max/Ctrl-brpts5-max.png';


opname = [Path Rootname '_NW_mean_junc_max.png'];

%junctions = imread(brname);
%beads = imread(brpts_fname);
%cluster = imread(thname);

mean_skel = imread(skname);
mean_junc = imread(juncname);

%rgb_junc = cat(3, junctions, junctions, junctions);
%rgb_beads = cat(3, beads, beads, beads);
%rgb_img = cat(3, img, img, img);
%rgb_cl = cat(3, cluster, cluster, cluster);

rgb_mean_skel = cat(3, mean_skel, mean_skel, mean_skel);
rgb_mean_junc = cat(3, mean_junc, mean_junc, mean_junc);

%rgb_img(:,:,1) = 0;
%rgb_img(:,:,3) = 0;

rgb_mean_skel(:,:,1) = 0;
rgb_mean_skel(:,:,3) = 0;
rgb_mean_junc(:,:,2) = 0;
%rgb_beads(:,:,2) = 0;
%rgb_junc(:,:,2) = 0;
%rgb_cl(:,:,2) = 0;

%overlay = imfuse(rgb_img, rgb_beads);
%overlay1 = imfuse(rgb_img, rgb_junc);
%overlay2 = imfuse(rgb_img, rgb_cl);
overlay = imfuse(rgb_mean_skel, rgb_mean_junc);

%mon = montage({rgb_img, rgb_beads, overlay}, 'size', [1 3], 'BorderSize', [1 1], 'BackgroundColor', 'white');
%mon = montage({rgb_img, rgb_junc, overlay1, rgb_cl, overlay2}, 'size', [1 5], 'BorderSize', [1 1], 'BackgroundColor', 'white');
mon = montage({overlay}, 'size', [1 1]);
mon_im = mon.CData;
imwrite(mon_im, opname);

function [Rootname, Path, Ext] = ExtractRootName(Name)

IdxP = strfind(Name,'.');
if ~isempty(IdxP)
    IdxP = IdxP(end);
    Root = Name(1:IdxP-1);
    Ext = Name(IdxP:end);
else
    Root = Name;
    Ext = '';
end

IdxSlash = strfind(Root,'\');
if ~isempty(IdxSlash)
    IdxSlash = IdxSlash(end);
    Path = Root(1:IdxSlash);
    Rootname = Root(IdxSlash+1:end);
else
    Path = '';
    Rootname = Root;
end


% runner for max porjection overlay
%for i=1:1
%name = ['/localhome/asa420/MIAL/aggregation-with-median/Climp/max/ClimpSeries' int2str(i) '-max.png'];
%bname = ['/localhome/asa420/MIAL/aggregation-with-median/Climp/Climp-brpts' int2str(i) '-max.png'];
%thname = ['/localhome/asa420/MIAL/aggregation-with-median/Climp/Climp-brpts' int2str(i) '-avg-th-norm.png'];
%sample_creator(name, bname, thname);
%end
