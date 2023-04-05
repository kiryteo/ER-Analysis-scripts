
function op = min_path(imname, n1, n2)
img = imread(imname);
sk = bwskel(img);
L = bwlabel(sk);
bw1 = (L==n1);
bw2 = (L==n2);
D1 = bwdist(bw1, 'quasi-euclidean');
D2 = bwdist(bw2, 'quasi-euclidean');
D = D1 + D2;
D = round(D * 32) / 32;
paths = imregionalmin(D);
paths_thinned_many = bwmorph(paths, 'thin', inf);

P = false(size(sk));
P = imoverlay(P, paths, [.5 .5 .5]);
P = imoverlay(P, paths_thinned_many, [1 1 0]);
P = imoverlay(P, sk, [1 1 1]);
imshow(P, 'InitialMagnification', 'fit');
