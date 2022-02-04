% Running loop for directories



rtndir = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn-beads/';
files = dir(fullfile(rtndir));

files

for k = 3:length(files)
    basefile = files(k).name;
    fullfname = fullfile(rtndir, basefile);
    ks_multithresh_beads(fullfname);
end
