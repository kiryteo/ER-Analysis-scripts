
function properties = ks_multithresh_beads(filename)

img = imread(filename);
conn_comp = bwconncomp(img);
labeled_conf = labelmatrix(conn_comp);
props = regionprops('table', labeled_conf, 'Area', 'Circularity', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'MajorAxisLength', 'MaxFeretProperties', 'MinFeretProperties', 'MinorAxisLength', 'Orientation', 'Perimeter');

series_name = split(filename, '/');
fname_ext = series_name(end);
fname = split(fname_ext, '.');
new_fname = strcat(fname(1), '.csv');

prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/climp-features/';

name = strcat(prefix, new_fname{1});
writetable(props, name);
