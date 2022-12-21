import glob
import imageio
from skimage.transform import resize

# pref = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/rtn_junc_viz/'


# pref = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junc_types_movies/'

pref = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_29_74/'

# with imageio.get_writer('A1_junc_iso_fuz.gif', mode='I', duration=0.5) as writer:
with imageio.get_writer('A1_iso_j15_egfp.gif', mode='I', duration=0.5) as writer:
# with imageio.get_writer('A1_junc.mp4', fps=2) as writer:
    for frame in range(100):
        filename = f'{pref}A1_decon_t0{frame:02d}_ch00.png'
        image = imageio.imread(filename)
        image = (image - image.min()) / (image.max() - image.min())
        # image = resize(image, (64, 68))
        writer.append_data(image)
