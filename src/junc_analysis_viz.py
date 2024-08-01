import os
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from skimage.measure import regionprops, find_contours

home = os.path.expanduser('~')
confocal_data_path = '/MIAL/data/confocal_movies/'

def read_image(path):
    return imageio.imread(path)

def save_image(path, img):
    imageio.imsave(path, img)

def get_junction_image(newps, shape=(128, 128)):
    brpts_img = np.zeros(shape)
    for each in newps:
        brpts_img[each[0], each[1]] = 255.
    return brpts_img

def junc_spread_comparison(group):
    img = read_image(f'{home}{confocal_data_path}{group}/er_mean_proc/atl1_er_mean_proc.png')
    im1 = read_image(f'{home}{confocal_data_path}{group}/junctions/A1_junc_mean.png')
    im2 = read_image(f'{home}{confocal_data_path}{group}/junctions/A1_proc_junc_mean.png')

    fig, ax = plt.subplots(1, 2)
    fig.suptitle('Junction detection methods comparison (based on input)', fontsize=12)

    ax[0].imshow(img)
    ax[0].imshow(im1)
    ax[0].set_title('Junction frame projection', fontsize=14)
    ax[0].axis('off')

    ax[1].imshow(img)
    ax[1].imshow(im2)
    ax[1].set_title('Junction based on mean projection of ER input frames', fontsize=14)
    ax[1].axis('off')

    plt.show()

def junc_spread_display(group, num_series):
    init_mean_proj_img = read_image(f'{confocal_data_path}{group}/junctions/{group[0]}{num_series}_junc_mean.png')
    junc_mean_proj_img = read_image(f'{confocal_data_path}{group}/junctions/{group[0]}{num_series}_proc_junc_mean.png')

    x1, y1 = np.where(init_mean_proj_img != 0)
    x2, y2 = np.where(junc_mean_proj_img != 0)

    img = read_image(f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc.png')
    plt.imshow(img)
    plt.plot(y1, x1, 'o', markerfacecolor='None', markeredgecolor='blue')
    plt.plot(y2, x2, 'o', markerfacecolor='None', markeredgecolor='red')
    plt.suptitle(f'{group} series {num_series}, Blue spots: junctions from a), Red spots: junctions from b)')
    plt.title('a) Projection from all junction frames b) Projection at the initial stage (ER input)')
    plt.show()

def per_frame_junc_projection(group, num_series):
    junc_mean = np.zeros((128, 128))
    for i in range(100):
        img = read_image(f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}/{group[0]}{num_series}_decon_t0{i:02d}_ch00_junc.png')
        junc_mean += img
    save_image(f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}_junc_mean.png', junc_mean / 100)

def init_proc_projection(group, num_series):
    sk = read_image(f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png')
    node_set, degree_list = skel_to_graph(sk)
    node_coords = np.array([node_set[node]['o'] for node in node_set])
    newps = [node_coords[j] for j, val in enumerate(degree_list) if val[1] > 2]
    brpts_img = get_junction_image(newps)
    save_image(f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}_proc_junc_mean.png', brpts_img)

def mean_frame_validation(group, total_series):
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
    pref = group_pref[group]
    for num_ser in range(1, total_series + 1):
        os.makedirs(f'{confocal_data_path}{group}/new_op_jul/junctions/{pref}{num_ser}', exist_ok=True)
        for frame in range(100):
            sk = read_image(f'{confocal_data_path}{group}/new_op_jul/skel/{pref}{num_ser}/{pref}{num_ser}_decon_t0{frame:02d}_ch00_skel.png')
            node_set, degree_list = skel_to_graph(sk)
            node_coords = np.array([node_set[node]['o'] for node in node_set])
            newps = [node_coords[j] for j, val in enumerate(degree_list) if val[1] > 2]
            brpts_img = get_junction_image(newps)
            save_image(f'{confocal_data_path}{group}/new_op_jul/junctions/{pref}{num_ser}/{pref}{num_ser}_decon_t0{frame:02d}_ch00_junc.png', brpts_img)

def junc_intensity_plot_creator(coord_dt):
    for k, v in coord_dt.items():
        plt.plot(v)
        plt.xlabel('Timeframe')
        plt.ylabel('Intensity values')
        plt.title(f"ATL 1, Junction coordinates: {k[1]['<unknown expression ERROR>', 0]} intensity variation")
        plt.savefig(f"ATL_1_Junction_{k[1]['<unknown expression ERROR>', 0]}.png", bbox_inches='tight', pad_inches=0.2)
        plt.close()

def viz_regionprops(labelled_img, spread_img):
    fig = px.imshow(spread_img, binary_string=True)
    fig.update_traces(hoverinfo='skip')

    props = regionprops(labelled_img, spread_img)
    properties = ['area', 'eccentricity', 'perimeter']

    for index in range(1, labelled_img.max()):
        label_i = props[index].label
        contour = find_contours(labelled_img == label_i)[0]
        y, x = contour.T
        hoverinfo = ''.join(f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>' for prop_name in properties)

        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)

def runner_viz_regionprops():
    img = read_image(f'{home}/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png')
    lab = read_image(f'{home}/ER-Analysis-scripts/Figure3-FuzIso/Climp12_junc_labelled_cc.png')
    img = np.stack((img, img, img, img), axis=2)
    print(img.shape)
    print(lab.shape)
    viz_regionprops(lab, img)

# Uncomment to run
# runner_viz_regionprops()