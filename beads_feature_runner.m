% Running loop for directories

beads_dir = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/climp-beads/';
files = dir(fullfile(beads_dir));

for k = 3:length(files)
    basefile = files(k).name;
    fullfname = fullfile(beads_dir, basefile);
    ks_multithresh_beads(fullfname);
end
