from PIL import Image
import imageio

import turtle
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


import matplotlib.pyplot as plt
from skimage import draw


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

prefix = '/localhome/asa420/Desktop/Climp/'


for i in range(1, 32):
	for j in range(100):
		bg = cv2.imread(prefix + 'skel/C%s/C%s_decon_t0%s_ch00_skel.png'%(f'{i}',f'{i}',f'{j:02d}'))
		fg = cv2.imread(prefix + 'brpts/C%s/C%s_decon_t0%s_ch00_skel_brpts.png'%(f'{i}',f'{i}',f'{j:02d}'))

		# bg = cv2.imread(prefix + 'skel/Ct%s/img_%s_decon_t0%s_skel.png'%(f'{i}',f'{i}',f'{j:02d}'))
		# fg = cv2.imread(prefix + 'brpts/Ct%s/img_%s_decon_t0%s_skel_brpts.png'%(f'{i}',f'{i}',f'{j:02d}'))

		junctions = np.where(fg[:,:,2]==255)
		skel = np.where(bg[:,:,2]==255)

		# rr, cc = draw.circle_perimeter(20, 33, radius=3, shape=arr.shape)
		# print(bg.shape)
		for x, y in zip(junctions[0], junctions[1]):
			rr, cc = draw.circle_perimeter(x, y, radius=2, shape=(128, 128))
			for a, b in zip(rr, cc):
				bg[a,b] = [0,255,255]

		# bg[rr, cc] = [255,0,255]
		# bg[skel] = [0,255,255]
		# fg[junctions] = [255,0,255]

		# bg[junctions] = [255,0,255]

		# op = Image.blend(Image.fromarray(bg), Image.fromarray(fg), 0.8)
		imageio.imwrite(prefix + 'overlay/C%s/C%s_decon_t0%s_ch00_skel_br_overlay.png'%(f'{i}',f'{i}',f'{j:02d}'), bg)
		# imageio.imwrite(prefix + 'overlay/Ct%s/img_%s_decon_t0%s_skel_br_overlay.png'%(f'{i}',f'{i}',f'{j:02d}'), bg)

exit()

bg = Image.open('/media/ashwin/Ashwin/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/enh/Series002_decon_converted_t00_ch00_std_enhance_skel.png').convert('RGBA')
fg = Image.open('/media/ashwin/Ashwin/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/brpts/Series002_decon_converted_t00_ch00_std_enhance_brpts.png').convert('RGBA')
op = Image.blend(bg, fg, 0.5)

imageio.imwrite('newov.png', op)

# import matplotlib.pyplot as plt
# plt.imshow(op)
# plt.show()

# import imageio
# imageio.imwrite('/home/ashwin/MIAL/blob-analysis/selection.png', op)
# import imageio

# img = imageio.imread('/home/ashwin/MIAL/blob-analysis/selection.png')
# import numpy as np

# print(np.unique(img))

# import copy
# new = copy.deepcopy(img)

# new[np.where(img==81)] = 220

# new[np.where(img==92)] = 255

# imageio.imwrite('/home/ashwin/MIAL/blob-analysis/selection-enh-2.png', new)



######## Overlay - fuse in matlab

# mask1 = abs(m(:,3)-m(:,1))<0.8;
# mask2 = abs(m(:,3)-m(:,2))>0.3;
# mask3 = abs(m(:,2)-m(:,1))>0.3;
# mask = mask1 & mask2 & mask3;
# m(mask,:) = 1-m(mask,:);
# rgb = ind2rgb(I,m);
