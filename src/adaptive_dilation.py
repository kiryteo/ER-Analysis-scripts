import os
import imageio
import numpy as np
from skimage import morphology

home = os.path.expanduser('~')
data_path = os.path.join(home, 'MIAL', 'data', 'sted-data', 'vess_enh_unet')

def create_adaptive_mask(group):
    for frame in range(1, 17):
        try:
            skel_path = os.path.join(data_path, group, 'skel', f'sted_{group}{frame}_proc_skel.png')
            skel = imageio.imread(skel_path)
            
            inv = np.logical_not(skel)
            dilskel = morphology.dilation(skel)
            closed_skel = morphology.area_closing(skel)
            roi = closed_skel - inv

            roi[roi == 255] = 0
            roi[roi == 254] = 255

            op = dilskel - roi
            save_path = os.path.join(data_path, group, 'adaptive_mask', f'{group}{frame}_er_mean_mask.png')
            imageio.imsave(save_path, op)
        except Exception as e:
            print(f"Error processing frame {frame} for group {group}: {e}")

if __name__ == "__main__":
    create_adaptive_mask('climp')