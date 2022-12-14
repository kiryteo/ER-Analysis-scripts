from PIL import Image
import imageio

import turtle
import cv2
import numpy as np

from matplotlib.patches import Circle
import copy

import matplotlib.pyplot as plt
from skimage import draw
from skimage.transform import resize



a = imageio.imread('/localhome/asa420/grp.png')
# b = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1/A1_decon_t000_ch00_skel.png')

b = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/A1_decon_t000_ch01_std_std_adj.png')

bst = np.stack((b,b,b), axis=2)

c = resize(a, (128, 128))

d = 0.5 * bst + 0.5 * c[:,:,:3]

# print(a.max())

plt.imshow(d)
plt.show()

exit()








def create_circles(junctions, overlay_image, radius):
    for x, y in zip(junctions[0], junctions[1]):
        rr, cc = draw.circle_perimeter(x, y, radius=radius, shape=(128, 128))
        for a, b in zip(rr, cc):
            overlay_image[a, b] = [0, 255, 255]
    return overlay_image


prefix = '/localhome/asa420/Desktop/RTN/'


adj_dir = '/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/'

gr_dir = '/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/'


# a = Image.open('/localhome/asa420/ER-Analysis-scripts/rerere.png')
# b = Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/A1_decon_t000_ch00_graph.png')

a = cv2.imread('/localhome/asa420/ER-Analysis-scripts/rerere.png')
b = cv2.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/A1_decon_t000_ch00_graph.png')



# bdata = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/A1_decon_t000_ch00_graph.png')

# b = resize(bdata, (509, 389))

# print(bdata.shape)

# a = np.array(a).astype('uint8')
# b = np.array(b).astype('uint8')

# print(a.dtype)
# print(a.dtype)

# a = Image.fromarray(a)
# b = Image.fromarray(b)

# print(a.format)
# print(b.format)

# print(a.size)
# print(b.size)

# op = Image.blend(a, b, 0.5)
# imageio.imwrite('gr_bl.png', op)


dst = cv2.addWeighted(a, 1, b, 0.5, 0.0)
# cv2.imshow('dst', dst)
cv2.imwrite('dst.png', dst)

exit()

def graph_overlay():
    for i in range(100):
        adj = Image.open(f'{adj_dir}A1_decon_t0{i:02d}_ch01_std_std_adj.png')
        gr = Image.open(f'{gr_dir}A1_decon_t0{i:02d}_ch00_graph.png')
        op = Image.blend(adj, gr, 0.5)
        imageio.imwrite(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph-overlay/A1_decon_t0{i:02d}_ch00_graph_overlay.png', op)

graph_overlay()
exit()


def create_overlay(total_series, num_frames):
    for series in range(1, total_series+1):
        for frame in range(num_frames):
            mch = cv2.imread(f'{prefix}mcherry/R{series}/R{series}_decon_t0{frame:02d}_ch01_std_adj.png')

            # skel = cv2.imread(
            #     prefix + 'skel/C%s/C%s_decon_t0%s_ch00_skel.png' \
			# 	% (f'{series}', f'{series}', f'{frame:02d}'))
            brpts = cv2.imread(f'{prefix}brpts/R{series}/R{series}_decon_t0{frame:02d}_ch00_skel_brpts.png')

            # skel = cv2.imread(prefix + 'C%s-mean.png')
            # brpts = cv2.imread(prefix + 'C%s-mean-brpts.png')

            junctions = np.where(brpts[:, :, 2] != 0)
            # network = np.where(skel[:, :, 2] != 0)

            overlay_image = copy.deepcopy(mch)
            overlay_op = create_circles(junctions, overlay_image, radius=1)

            imageio.imwrite(f'{prefix}mch_junc_overlay/R{series}/R{series}_decon_t0{frame:02d}_ch00_mch_junc_overlay.png', overlay_op)
            # imageio.imwrite(prefix + 'C%s-NW_mean_junc_mean.png', overlay_op)

create_overlay(29, 100)

exit()

def create_overlay_br(total_series):
    for series in range(1, total_series):
        skel = cv2.imread(f'{prefix}C{series}-mean.png')
        brpts = cv2.imread(f'{prefix}C{series}-max-brpts.png')

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

        imageio.imwrite(f'{prefix}C{series}-NW_mean_junc_max.png', skel)

#create_overlay_br(32)

#exit()

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
