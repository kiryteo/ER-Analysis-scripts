from PIL import Image
import imageio

import turtle
import cv2
import numpy as np

from matplotlib.patches import Circle
import copy

import matplotlib.pyplot as plt
from skimage import draw


def create_circles(junctions, overlay_image, radius):
    for x, y in zip(junctions[0], junctions[1]):
        rr, cc = draw.circle_perimeter(x, y, radius=radius, shape=(128, 128))
        for a, b in zip(rr, cc):
            overlay_image[a, b] = [0, 255, 255]
    return overlay_image


prefix = '/localhome/asa420/Desktop/Climp/'

# /localhome/asa420/Desktop/Climp/C3-mean.png


def create_overlay(total_series, num_frames):
    for series in range(1, total_series):
        for frame in range(num_frames):
            # skel = cv2.imread(
            #     prefix + 'skel/C%s/C%s_decon_t0%s_ch00_skel.png' \
			# 	% (f'{series}', f'{series}', f'{frame:02d}'))
            # brpts = cv2.imread(
            #     prefix + 'brpts/C%s/C%s_decon_t0%s_ch00_skel_brpts.png' \
			# 	% (f'{series}', f'{series}', f'{frame:02d}'))
            skel = cv2.imread(prefix + 'C%s-mean.png')
            brpts = cv2.imread(prefix + 'C%s-mean-brpts.png')

            junctions = np.where(brpts[:, :, 2] == 255)
            skel = np.where(skel[:, :, 2] == 255)

            overlay_image = copy.deepcopy(skel)

            # rr, cc = draw.circle_perimeter(20, 33, radius=3, shape=arr.shape)
            # print(bg.shape)
            # for x, y in zip(junctions[0], junctions[1]):
            # 	rr, cc = draw.circle_perimeter(x, y, radius=2, shape=(128, 128))
            # 	for a, b in zip(rr, cc):
            # 		skel[a,b] = [0,255,255]
            overlay_op = create_circles(junctions, overlay_image, radius)

            # imageio.imwrite(prefix + 'overlay/C%s/C%s_decon_t0%s_ch00_skel_br_overlay.png' % (
            #     f'{series}', f'{series}', f'{frame:02d}'), overlay_op)
            imageio.imwrite(prefix + 'C%s-NW_mean_junc_mean.png', overlay_op)

def create_overlay_br(total_series):
    for series in range(1, total_series):
        skel = cv2.imread(prefix + 'C%s-mean.png' % f'{series}')
        brpts = cv2.imread(prefix + 'C%s-max-brpts.png' % f'{series}')

        # junctions = np.where(brpts[:, :, 2] == 255)
        junctions = np.where(brpts[:, :, 2] != 0)
        # skeleton = np.where(skel[:, :, 2] == 255)

        # overlay_image = copy.deepcopy(skel)
        # print(junctions)

        # rr, cc = draw.circle_perimeter(20, 33, radius=3, shape=arr.shape)
        # print(bg.shape)

        # for x, y in zip(junctions[0], junctions[1]):
        #     rr, cc = draw.circle_perimeter(x, y, radius=1, shape=(128, 128))
        #     for a, b in zip(rr, cc):
        #         skel[a,b] = [0,255,255]

        for x, y in zip(junctions[0], junctions[1]):
            # rr, cc = draw.circle_perimeter(x, y, radius=1, shape=(128, 128))
            skel[x, y] = [0,255,255]
            # for a, b in zip(rr, cc):
            #     skel[a,b] = [0,255,255]

        # print(skel.shape)

        imageio.imwrite(prefix + 'C%s-NW_mean_junc_max.png' % f'{series}', skel)

create_overlay_br(32)

exit()

def pil_blend():

    bg = Image.open('/localhome/asa420/Desktop/ATL/A1-mean.png')
    fg = Image.open('/localhome/asa420/Desktop/ATL/A1-mean-brpts.png')
    # bg = Image.open('/media/ashwin/Ashwin/live-cell-movies/COSKDEL/COSKDEL/Decon/\
	# 				Series002_decon_converted/enh/Series002_decon_converted_t00_\
	# 				ch00_std_enhance_skel.png').convert('RGBA')
    # fg = Image.open('/media/ashwin/Ashwin/live-cell-movies/COSKDEL/COSKDEL/Decon/\
	# 				Series002_decon_converted/brpts/Series002_decon_converted_t00\
	# 				_ch00_std_enhance_brpts.png').convert('RGBA')

    op = Image.blend(bg, fg, 0.5)
    imageio.imwrite('newovee.png', op)


######## Overlay - fuse in matlab

# mask1 = abs(m(:,3)-m(:,1))<0.8;
# mask2 = abs(m(:,3)-m(:,2))>0.3;
# mask3 = abs(m(:,2)-m(:,1))>0.3;
# mask = mask1 & mask2 & mask3;
# m(mask,:) = 1-m(mask,:);
# rgb = ind2rgb(I,m);


# arr = np.zeros((200, 200))
# rr, cc = draw.circle_perimeter(20, 33, radius=3, shape=arr.shape)
# arr[rr, cc] = 1
# plt.imshow(arr)
# plt.show()

# exit()

# t = turtle.Turtle()

# def drawcirc(x,y,r):
# 	t.pu()
#     t.goto(x,y-r) #-r because we want xy as center and Turtles starts from border
#     t.pd()
#     t.circle(r)
