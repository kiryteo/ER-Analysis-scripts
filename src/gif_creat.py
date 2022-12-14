import glob
import imageio

# pref = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/rtn_junc_viz/'


pref = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junc_types_movies/'

with imageio.get_writer('A1_junc_iso_fuz.gif', mode='I', duration=0.5) as writer:
# with imageio.get_writer('A1_junc.mp4', fps=2) as writer:
    for i in range(100):
        filename = f'{pref}A1_decon_t0{i:02d}_ch00.png'
        image = imageio.imread(filename)
        image = (image - image.min()) / (image.max() - image.min())
        writer.append_data(image)
