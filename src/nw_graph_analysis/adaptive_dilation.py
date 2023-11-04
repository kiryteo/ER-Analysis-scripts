import contextlib
import imageio
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt


# img = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/climp/images/climp4_er_mean.png')
# skel = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/climp/skel/climp4_proc_skel.png')

# inv = np.logical_not(skel)
# # distance = ndi.distance_transform_edt(inv)
# # labels = morphology.label(distance)
# # invlab = morphology.label(inv, connectivity=1)
# # bin_invlab = (invlab > 0).astype(np.uint8)

# dilskel = morphology.dilation(skel)
# # roi = morphology.area_closing(skel) - bin_invlab
# closed_skel = morphology.area_closing(skel)
# roi = closed_skel - inv

# roi[np.where(roi==255)] = 0.
# roi[np.where(roi==254)] = 255.

# op = dilskel - roi


def create_adaptive_mask(group):
    for frame in range(1, 17):
        try:
            # skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/{group}/skel/{group}{frame}_proc_skel.png')
            skel = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/skel/sted_{group}{frame}_proc_skel.png')
            inv = np.logical_not(skel)
            dilskel = morphology.dilation(skel)
            closed_skel = morphology.area_closing(skel)
            roi = closed_skel - inv

            roi[np.where(roi==255)] = 0.
            roi[np.where(roi==254)] = 255.

            op = dilskel - roi
            imageio.imsave(f'/localhome/asa420/MIAL/data/sted-data/vess_enh_unet/{group}/adaptive_mask/{group}{frame}_er_mean_mask.png', op)
            # imageio.imsave(f'/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/{group}/adaptive_mask/{group}{frame}_er_mean_mask.png', op)
        except:
            pass

create_adaptive_mask('climp')

exit()

plt.figure(figsize=(16,12))


plt.subplot(231)
plt.imshow(skel, cmap='gray')
plt.title('Skeleton')
plt.axis('off')

plt.subplot(232)
plt.imshow(inv, cmap='gray')
plt.title('Inverted Skeleton')
plt.axis('off')

plt.subplot(233)
plt.imshow(closed_skel, cmap='gray')
plt.title('Area closing')
plt.axis('off')

plt.subplot(234)
plt.imshow(roi, cmap='gray')
plt.title('Close proximity regions')
plt.axis('off')

plt.subplot(235)
plt.imshow(dilskel, cmap='gray')
plt.title('Dilated skeleton')
plt.axis('off')

plt.subplot(236)
plt.imshow(op, cmap='gray')
plt.title('Updated mask')
plt.axis('off')





# plt.subplot(241)
# plt.imshow(inv, cmap='gray')
# plt.title('Inverted Skeleton')
# plt.axis('off')


# plt.subplot(242)
# plt.imshow(distance, cmap='gray')
# plt.title('EDT')
# plt.axis('off')

# # plt.subplot(243)
# # plt.imshow(labels, cmap='gray')
# # plt.title('Conn. components')
# # plt.axis('off')


# plt.subplot(243)
# plt.imshow(invlab, cmap='gray')
# plt.title('Inverted Skel labels')
# plt.axis('off')

# plt.subplot(244)
# plt.imshow(bin_invlab, cmap='gray')
# plt.title('Binary inverted labels')
# plt.axis('off')

# plt.subplot(245)
# plt.imshow(morphology.area_closing(skel), cmap='gray')
# plt.title('Area closing')
# plt.axis('off')

# plt.subplot(246)
# plt.imshow(roi, cmap='gray')
# plt.title('Close proximity regions')
# plt.axis('off')

# plt.subplot(247)
# plt.imshow(dilskel, cmap='gray')
# plt.title('Dilated skeleton')
# plt.axis('off')

# plt.subplot(248)
# plt.imshow(op, cmap='gray')
# plt.title('Updated mask')
# plt.axis('off')

# plt.subplot(154)
# plt.imshow(dilskel)
# plt.title('Dilated skeleton')
# plt.axis('off')


# plt.subplot(155)
# # plt.imshow(bin_invlab, cmap='gray')
# plt.imshow(op)
# plt.title('Updated mask')
# plt.axis('off')

# plt.show()

plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig('updated_mask_v3.png', bbox_inches='tight', pad_inches=0.02)
plt.close()