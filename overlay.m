im1 = imread('');
im2 = imread('');
im3 = imread('');

rgbim1 = cat(3, im1, im1, im1);
rgbim2 = cat(3, im2, im2, im2);
rgbim3 = cat(3, im3, im3, im3);

rgbim1(:,:,1) = 0;
rgbim2(:,:,2) = 0;

l = imfuse(rgbim1, rgbim2);
m = imfuse(im3, l);

imwrite(m, '');
