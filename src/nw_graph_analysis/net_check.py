import contextlib
import numpy as np
from skimage.measure import label, regionprops
import itertools
import sknw
import imageio
import matplotlib.pyplot as plt
import copy
from skimage.morphology import skeletonize, erosion, binary_erosion, medial_axis
# import graph_connector_modules as gcm
import skimage
from skimage.filters import threshold_local, threshold_otsu
import cv2
from PIL import Image

# def create_tiff_stack(images, output_path):
#     # Create a new blank image to store the stack
#     stack = Image.new('L', images[0].size)

#     # Create the TIFF stack by adding each image as a frame
#     for image in images:
#         stack.save(output_path, append=True)
#         gr_image = image.convert('L')
#         stack.paste(gr_image)

#     # Save the final stack
#     stack.save(output_path)

# # Example usage
# # input_images = [
# #     Image.open('image1.jpg'),
# #     Image.open('image2.jpg'),
# #     Image.open('image3.jpg'),
# # ]

# input_images = [
#     Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A18_decon_t027_ch00_std.png'),
#     Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A18_decon_t028_ch00_std.png'),
#     Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A18_decon_t029_ch00_std.png'),
#     Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A18_decon_t030_ch00_std.png'),
#     Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A18_decon_t031_ch00_std.png'),
# ]

# output_file = 'stacked_images.tif'

# create_tiff_stack(input_images, output_file)


# exit()

img = imageio.imread('/localhome/asa420/Downloads/ERnet-v2-main/Training/Simulate_ER/ER_250_20230707/0_GT.png')#0_IN_snr8.49.png')

aop = skimage.morphology.area_opening(img, area_threshold=2)
erod = skimage.morphology.erosion(aop)
aop = skimage.morphology.area_opening(erod, area_threshold=2)
cl = skimage.morphology.area_closing(aop, area_threshold=32)
aop = skimage.morphology.area_opening(cl, area_threshold=2)

# cv2.imwrite('morph-ops_c12_t0.png', aop)


loc = threshold_local(aop, 3)
loc = threshold_local(loc, 3)

# thr = threshold_otsu(loc)

bloc = (loc > 0)

sk = skeletonize(bloc)

plt.imshow(sk)
plt.show()
exit()



# input_path = '/localhome/asa420/Downloads/ERnet-v2-main/Synthetic ER dataset for connectivity test_1/Synthetic ER dataset for connectivity test_1/ER_1000_20221030_noise_0.0/0_GT_noise_0.png'
input_path = '/localhome/asa420/Downloads/ERnet-v2-main/Training/Simulate_ER/ER_250_20230707/0_IN_snr8.49.png'

img = imageio.imread(input_path)
print(img.shape)

print(np.unique(img))

img = erosion(img)

ip = (img > 0).astype(np.uint8)

ip = binary_erosion(ip)
ip = binary_erosion(ip)

sk = skeletonize(ip)
# sk = medial_axis(ip)

# plt.imshow(sk, cmap='gray')
# plt.show()

gr = sknw.build_sknw(sk, multi=True, iso=False)


plt.imshow(ip, cmap='gray')

for (s,e) in gr.edges():
    ps = gr[s][e][0]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')
    with contextlib.suppress(Exception):
        ps = gr[s][e][1]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')

# draw node by o
nodes = gr.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')


plt.show()

exit()