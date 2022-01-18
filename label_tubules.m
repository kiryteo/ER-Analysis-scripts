function tubules = label_tubules(filename)

[Rootname,Path,~] = ExtractRootName(filename);

output_filename = [Path Rootname '_labeled.png'];

img = imread(filename);
bin_img = imbinarize(img);

op = bwconncomp(bin_img);
l = labelmatrix(op);

imwrite(l, output_filename, 'png');


function [RootName,Path,Ext] = ExtractRootName(Name)
%% Function which extracts the extension (Ext), the folder name (Path) and the rootname from a string name of a file

% Find point and extract extension
IdxP = strfind(Name,'.');
if ~isempty(IdxP)
    IdxP = IdxP(end);
    Root = Name(1:IdxP-1);
    Ext = Name(IdxP:end);
else
    Root = Name;
    Ext = '';
end

% Find backslash and extract folder name
IdxSlash = strfind(Root,'\');
if ~isempty(IdxSlash)
    IdxSlash = IdxSlash(end);
    Path = Root(1:IdxSlash);
    RootName = Root(IdxSlash+1:end);
else
    Path = '';
    RootName = Root;
end