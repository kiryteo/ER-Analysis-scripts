import os
import imageio
from skimage.io import imread
import matplotlib.pyplot as plt

home = os.path.expanduser('~')

# STED = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27.npy-1205-1702-Synth-STED.png')
# KM_op = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27_entropyKapurMultiTh.jpg')
# SM_op = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27_singh.jpg')
# beads_op = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27_beads.png')


# STED = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/sted/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27.npy-1205-1702-STED.png')
# KM_op = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/sted/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27_entropyKapurMultiTh.jpg')
# SM_op = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/sted/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27_singh.jpg')
# beads_op = imread('/localhome/asa420/MIAL/blob-analysis/thresholds/op-new/sted/3_23_2021_RTN4ACOS7_Paired_STED_Decon_Series006_decon_ch02_densePER__x1205_y1702_coverage27_beads.png')


STED = imread(home + '/MIAL/blob-analysis/1_Series012_decon_merged_patch10_sted.png')
KM_op = imread(home + '/MIAL/blob-analysis/1_Series012_decon_merged_patch10_sted_entropyKapurMultiTh.jpg')
SM_op = imread(home + '/MIAL/blob-analysis/1_Series012_decon_merged_patch10_sted_singh.jpg')
beads_op = imread(home + '/MIAL/blob-analysis/1_Series012_decon_merged_patch10_sted_beads.png')

fig, ax = plt.subplots(1, 4, figsize=(10, 6))

ax[0].imshow(STED)
ax[0].set_title('STED')

ax[1].imshow(SM_op)
ax[1].set_title('local thresholding')

ax[2].imshow(KM_op)
ax[2].set_title('global thresholding')

ax[3].imshow(beads_op)
ax[3].set_title('Extracted beads')

fig.tight_layout()
# plt.subplots_adjust(wspace=1, hspace=1)
plt.savefig("Ctrl-STED-sample.png", bbox_inches='tight', dpi=1000)
# plt.show()
