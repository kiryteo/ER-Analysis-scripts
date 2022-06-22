
fname = ['/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/Series006_decon_ch02-3'];
tub = imread([fname '_tubules.png']);

brpts = imread([fname '_dilbr.png']);

f = imfuse(tub, brpts);
imwrite(f, [fname '_skel_br.png']);
