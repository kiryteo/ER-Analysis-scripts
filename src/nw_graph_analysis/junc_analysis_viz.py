import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt


confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'

# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure2/A1_decon_t000_ch00_skel.png')
# plt.axis('off')
# plt.imshow(img, cmap='gray')
#
# plt.savefig('A1_t0_skel_dpi700.png', bbox_inches='tight', pad_inches=0, dpi=700)
# plt.close()
# # plt.show()
# exit()


def junc_spread_comparison():
    """
    Compare the junctions obtained via initial projection Vs.
    per frame junction projection
    @return:
    """
    img = imageio.imread(f'{confocal_data_path}ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')
    im1 = imageio.imread(f'{confocal_data_path}ATL/new_op_jul/junctions/A1_junc_mean.png')
    im2 = imageio.imread(f'{confocal_data_path}ATL/new_op_jul/junctions/A1_proc_junc_mean.png')

    fig, ax = plt.subplots()
    r, c = 1, 2

    plt.axis('off')
    fig.add_subplot(r, c, 1)
    plt.title('Junction frame projection', fontsize=14)
    plt.imshow(img)
    plt.imshow(im1)

    fig.add_subplot(r, c, 2)
    plt.title('Junction based on mean projection of ER input frames', fontsize=14)
    plt.imshow(img)
    plt.imshow(im2)

    # plt.title('Junction detection methods comparison (based on input)', fontsize=12)
    plt.show()


def junc_spread_display(group, num_series):
    """

    @param group:
    @param num_series:
    @return: display mean proj frame junctions (red spots) + per frame skel junctions (blue spots)
    """
    init_mean_proj_img = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}_junc_mean.png')

    junc_mean_proj_img = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}_proc_junc_mean.png'))

    init_proj_img_coords = np.where(init_mean_proj_img != 0)
    junc_proj_img_coords = np.where(junc_mean_proj_img != 0)

    x1, y1 = init_proj_img_coords[0], init_proj_img_coords[1]
    x2, y2 = junc_proj_img_coords[0], junc_proj_img_coords[1]

    fl = imageio.imread('R1_opflow.png')
    stable_pts = np.where(fl == 255)
    moving_pts = np.where()

    # n1 = []
    # for a,b in zip(x1, y1):
    #     n1.append([a, b])
    #
    # n2 = []
    # for a,b in zip(x2, y2):
    #     n2.append([a, b])

    # print(n1)
    # print(n2)
    img = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc.png'))

    plt.imshow(img)
    # plt.scatter(y1, x1, color='red')
    # plt.scatter(y2, x2, color='blue')

    plt.plot(y1, x1, 'o', markerfacecolor='None', markeredgecolor='blue')
    plt.plot(y2, x2, 'o', markerfacecolor='None', markeredgecolor='red')

    plt.suptitle(f'{group} series {num_series}, Blue spots: junctions from a), Red spots: junctions from b)')

    plt.title('a) Projection from all junction frames b) Projection at the initial stage (ER input)')
    plt.show()

    # plt.plot(im1, 'r.')
    # plt.plot(im2, 'b.')
    # plt.show()


def get_junction_image(newps):
    """

    @param newps: List of nodes
    @return: Image with nodes -> 1, else 0
    """
    brpts_img = np.zeros((128, 128))
    for each in newps:
        brpts_img[each[0], each[1]] = 255.
    return brpts_img


def per_frame_junc_projection(group, num_series):
    junc_mean = np.zeros((128, 128))
    for i in range(100):
        img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}/{group[0]}{num_series}_decon_t0{i:02d}_ch00_junc.png'))

        junc_mean += img

    cv2.imwrite((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}_junc_mean.png'), junc_mean / 100)

def init_proc_projection(group, num_series):
    sk = imageio.imread((confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'))


    node_set, degree_list = skel_to_graph(sk)
    node_coords = np.array([node_set[node]['o'] for node in node_set])

    # get all the nodes with degree greater than 2
    newps = [node_coords[j] for j, val in enumerate(degree_list) if val[1] > 2]
    brpts_img = get_junction_image(newps)

    cv2.imwrite((confocal_data_path + f'{group}/new_op_jul/junctions/{group[0]}{num_series}_proc_junc_mean.png'), brpts_img)


def mean_frame_validation(group, total_series):
    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for num_ser in range(1, total_series + 1):
        os.makedirs(confocal_data_path + f'{group}/new_op_jul/junctions/{group_pref[group]}{num_ser}')

        for frame in range(100):
            sk = imageio.imread((confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_ser}/{group_pref[group]}{num_ser}_decon_t0{frame:02d}_ch00_skel.png'))


            node_set, degree_list = skel_to_graph(sk)
            node_coords = np.array([node_set[node]['o'] for node in node_set])

            # get all the nodes with degree greater than 2
            newps = [node_coords[j] for j, val in enumerate(degree_list) if val[1] > 2]
            brpts_img = get_junction_image(newps)

            imageio.imsave((confocal_data_path + f'{group}/new_op_jul/junctions/{pref}{num_ser}/{pref}{num_ser}_decon_t0{frame:02d}_ch00_junc.png'), brpts_img)


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

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labelled_img.max()):
        label_i = props[index].label
        contour = measure.find_contours(labelled_img == label_i)[0]
        y, x = contour.T
        hoverinfo = ''.join(f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>' for prop_name in properties)

        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)

def runner_viz_regionprops():
    img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png')
    # lab = label(img)
    lab = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/Climp12_junc_labelled_cc.png')

    img = np.stack((img, img, img, img), axis=2)

    print(img.shape)
    print(lab.shape)

    viz_regionprops(lab, img)

# runner_viz_regionprops()
# exit()