import shutil
import os
from PIL import Image
import numpy as np


def data_mover(group):
    prefix = f'/localhome/asa420/MIAL/data/classification_data/{group}/'

    group_pref = {'Atlastin': 'A', 'Climp': 'C', 'Control': 'Ct', 'Reticulon': 'R'}

    group_num = {'Atlastin': 26, 'Climp': 31, 'Control': 31, 'Reticulon': 29}

    for movie_num in range(1, group_num[group]+1):
        os.makedirs(f"{prefix}{movie_num}", exist_ok=True)
        for frame in range(100):
            shutil.move(f'{prefix}{group_pref[group]}{movie_num}_decon_t{frame:03d}_ch00_std.png', f'{prefix}{movie_num}/{group_pref[group]}{movie_num}_decon_t{frame:03d}_ch00_std.png')

# data_mover('Atlastin')
# data_mover('Climp')
# data_mover('Control')
# data_mover('Reticulon')
            
# def create_3d_vols(group):
#     prefix = f'/localhome/asa420/MIAL/data/classification_data/{group}/'

#     group_pref = {'Atlastin': 'A', 'Climp': 'C', 'Control': 'Ct', 'Reticulon': 'R'}

#     group_num = {'Atlastin': 26, 'Climp': 31, 'Control': 31, 'Reticulon': 29}

#     # iterate through all movies and create a volume of 100 frames each

#     for movie_num in range(1, group_num[group]+1):
#         frames = []
#         for frame in range(100):
#             frames.append(Image.open(f'{prefix}{movie_num}/{group_pref[group]}{movie_num}_decon_t{frame:03d}_ch00_std.png'))
#         volume = np.stack([np.array(frame) for frame in frames], axis=0)
#         np.save(f'{prefix}{movie_num}/volume.npy', volume)

# import nibabel as nib
# import numpy as np

# img = nib.load('/localhome/asa420/Documents/ER-Analysis-scripts/src/output.nii.gz')

# Nifti_img  = img
# nii_data = img.get_fdata()
# nii_aff  = img.affine
# nii_hdr  = img.header
# # print(nii_aff ,'\n',nii_hdr)
# print(nii_data.shape)


# exit()




import nibabel as nib
import numpy as np
import imageio

def nib_creator():
    # Assuming your frames are stored in a list called 'frames'
    # Each frame should be a 2D numpy array (128x128)

    group = 'Reticulon'

    for num in range(1, 32):
        frames = []

        for frame in range(100):
            frames.append(imageio.imread(f'/localhome/asa420/MIAL/data/classification_data/{group}/{num}/R{num}_decon_t0{frame:02d}_ch00_std.png'))

        # Generate a 3D array by stacking the frames along the third dimension
        data = np.stack(frames, axis=-1)

        # Create a NIfTI image object
        nifti_img = nib.Nifti1Image(data, affine=np.eye(4))  # You might need to adjust the affine matrix

        # Save the NIfTI image to a file (e.g., 'output.nii.gz')
        nib.save(nifti_img, f'/localhome/asa420/MIAL/data/classification_data/{group}/rtn{num}.nii.gz')

