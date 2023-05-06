import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)

# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure2/A1_decon_t000_ch00_skel.png')
# plt.axis('off')
# plt.imshow(img, cmap='gray')
#
# plt.savefig('A1_t0_skel_dpi700.png', bbox_inches='tight', pad_inches=0, dpi=700)
# plt.close()
# # plt.show()
# exit()

def junction_crops_creator(group, ser_num, junc_id, channel):
    nps, skdata, labelled_img = junc_analysis.label_junctions(group, ser_num)

    label_vals, unassigned_cc_dict = junc_analysis.separate_junc_cc(nps, skdata, labelled_img)
    iso, fuz, unk = junc_analysis.get_junction_areas(label_vals, unassigned_cc_dict)

    iso_cc = get_cc_ids(labelled_img, iso)
    iso_cc_coords = {each: np.where(labelled_img == each) for each in iso_cc}

    if channel == 'egfp':
        ch = 0
    else:
        ch = 1

    for i in range(100):
        file = get_std_img(
            f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{ser_num}_decon_t0{i:02d}_ch0{ch}.tif')

        # file_mch = get_std_img(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{ser_num}_decon_t0{i:02d}_ch01.tif')

        # file = get_std_img(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_29_74_CC/A{ser_num}_decon_t0{i:02d}_ch00.png')

        regions = regionprops(labelled_img)
        # print(regions[junc_id].label)
        #
        # exit()

        x, y = iso[junc_id][0], iso[junc_id][1]

        # contour code
        j = get_label_id(regions, iso, junc_id)
        label_i = regions[j].label

        contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
        cntrY, cntrX = contour.T

        # crp_egfp = file_egfp[x-5:x+5, y-5:y+5]
        # crp_mch = file_mch[x-5:x+5, y-5:y+5]

        plt.axis('off')
        # plt.imshow(crp_file, cmap='gray')

        plt.imshow(file, cmap='gray', interpolation=None)
        plt.plot(cntrX, cntrY, color='red', linewidth=0.0001)

        # fig = plt.gca()
        # crp = fig[x-5:x+5, y-5:y+5]

        # plt.imshow(crp_file)
        # plt.title(f't={i}')
        # fig = plt.gca(figsize=(8,10))

        plt.savefig(
            f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junction_crops/S{ser_num}_j{junc_id}_{x}_{y}_CC_crops/{group[0]}{ser_num}_decon_t0{i:02d}_ch0{ch}.png',
            bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()

        # imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junction_crops/S{ser_num}_junc{junc_id}_{x}_{y}/{group[0]}{ser_num}_decon_t0{i:02d}_ch00.png', crp_egfp)
        # imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/junction_crops/S{ser_num}_junc{junc_id}_{x}_{y}/{group[0]}{ser_num}_decon_t0{i:02d}_ch01.png', crp_mch)


# junction_crops_creator('ATL', 1, 15, 'mCherry')


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

# junc_spread_comparison()
# exit()


def junc_spread_display(group, num_series):
    """

    @param group:
    @param num_series:
    @return: display mean proj frame junctions (red spots) + per frame skel junctions (blue spots)
    """
    init_mean_proj_img = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}_junc_mean.png')

    junc_mean_proj_img = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/junctions/{group[0]}{num_series}_proc_junc_mean.png')

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
        f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc.png')

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

def crop_img():
    mean_img = confocal_data_path + 'ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)
    for i in range(100):
        img = imageio.imread(confocal_data_path + f'ATL/new_op_jul/A1_junc_viz/j48_skel/ATL1_t{i:02d}.png')

        # plt.imshow(img)
        # plt.show()

        # y = nps[74,1]
        # x = nps[74,0]
        # #
        # # # print(y, x)
        cimg = img[137 - 30:137 + 30, 143 - 30:143 + 30]

        plt.axis('off')
        plt.title(f't={i}')
        plt.imshow(cimg, interpolation='nearest', aspect='auto')
        plt.savefig(confocal_data_path + f'ATL/new_op_jul/A1_junc_viz/j48_skel/crops/ATL1_t{i:02d}.png',
                    bbox_inches='tight', pad_inches=0)

        # imageio.imsave('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/j47_skel/crops/ATL1_t%s.png'%f'{i:02d}', cimg)
        # plt.close()
        plt.show()

# def plot_junc_areas_og(group, series_num, labelled_img, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords):
def plot_junc_areas_og(group, series_num, labelled_img, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords):
    for i in range(1):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(f'{confocal_data_path}{group}/files/img_{series_num}_decon_t0{i:02d}.tif')
        else:
            img = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())

        # resc_img = skimage.transform.rescale(img, 2, anti_aliasing=False)

        # plt.imshow(resc_img, cmap='gray', interpolation=None)

        plt.imshow(img, cmap='gray', interpolation=None)

        # plt.plot(iso[:, 1], iso[:, 0], 's', markerfacecolor='None', markeredgecolor='red')

        # plt.plot(skdata[:, 1], skdata[:, 0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        for k, v in iso_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='red', mew=0.4)
        #
        for k, v in fuz_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='blue', mew=0.4)
        #
        # for k, v in unk_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        #
        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='white', mew=0.6)
        #
        # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)

        #######################################

        # for k, v in iso_cc_coords.items():
        #     plt.plot(v[1], v[0], 's', markerfacecolor='magenta', markeredgecolor='magenta', mew=0.35, ms=20)
        #
        # for k, v in fuz_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='blue', mew=0.35)
        #
        # for k, v in unk_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.5)
        #
        # if len(fuz) > 0:
        #     plt.plot(fuz[:, 1], fuz[:, 0], '.', markerfacecolor='None', markeredgecolor='white', mew=0.6)

        # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)
        # plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)

        # plots contours
        # for index in range(1, labelled_img.max()):
        #     label_i = regions[index].label
        #     contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
        #     y, x = contour.T
        #     plt.plot(x, y, color='cyan')

        # for index in range(1, op.max()):
        #     label_i = regions[index].label
        #     contour = measure.find_contours(op == label_i, 0.8)[0]
        #     y, x = contour.T
        #     plt.plot(x, y, color='cyan')

        # plt.axis('off')
        # plt.savefig('Climp_series12_junc_representation_iso_fuz_unk_2_new_colors', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.close()

        plt.show()


def plot_ref_iso_fuz_junc(group, series_num, iso, fuz):
    for i in range(1):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(f'{confocal_data_path}{group}/files/img_{series_num}_decon_t0{i:02d}.tif')
        else:
            img = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img, cmap='gray', interpolation=None)

        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], '.', markerfacecolor='None', markeredgecolor='blue', mew=1)

        plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='red', mew=1)

        plt.axis('off')
        plt.savefig(f'{group}_series{series_num}_ref_iso_fuz', bbox_inches='tight', pad_inches=0, dpi=700)
        plt.close()

        # plt.show()


def get_cc_ids(labelled_img, region):
    """
    Get CC ids for the specified region
    @param labelled_img: labelled image
    @param region: isolated or fuzzy
    @return: list of CC ids
    """

    # Create a dictionary to store per component data
    cc_data = {cc_id: [] for cc_id in np.unique(labelled_img)}

    # Populate the dictionary with locations for the specified region
    for loc in region:
        loc_x, loc_y = loc[0], loc[1]
        cc_id = labelled_img[loc_x, loc_y]
        cc_data[cc_id].append(loc)

    # Extract CC ids for the specified region
    cc_ids = [cc_id for cc_id, data in cc_data.items() if cc_id > 0 and len(data) > 0]

    return cc_ids


ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions('Control', 13)
# dict with ids as key and (x, y) as value
label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

# iso, fuz, unk: list of lists with x, y
iso, fuz, unk = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)

# iso_cc = get_cc_ids(labelled_img, iso)
# fuz_cc = get_cc_ids(labelled_img, fuz)
# # unk_cc = get_cc_ids(labelled_img, unk)

# iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
# fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}
# unk_cc_coords = {each: np.where(labelled_img==each) for each in unk_cc}
#
#
# iso_list = []
# for each in iso:
#     iso_list.append([each[0], each[1]])

# plot_junc_areas_og('Climp', 12, labelled_img, iso, fuz, per_frame_junctions, iso_cc_coords, fuz_cc_coords)

plot_ref_iso_fuz_junc('Control', 13, iso, fuz)

exit()



# def plot_junc_areas(group, series_num, iso, fuz, unk, labelled_img):
def plot_junc_areas(group, series_num, labelled_img, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords):
    regions = regionprops(labelled_img)
    # regions = regionprops(op)

    for i in range(1):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(f'{confocal_data_path}{group}/files/img_{series_num}_decon_t0{i:02d}.tif')
        else:
            img = imageio.imread(f'{confocal_data_path}{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())

        resc_img = skimage.transform.rescale(img, 2, anti_aliasing=False)

        plt.imshow(resc_img, cmap='gray', interpolation=None)
        # plt.plot(iso[:, 1], iso[:, 0], 's', markerfacecolor='None', markeredgecolor='red')

        # plt.plot(skdata[:, 1], skdata[:, 0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        # for k, v in iso_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        #
        # for k, v in fuz_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.4)
        #
        # for k, v in unk_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', m
        for k, v in fuz_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='blue', mew=0.35)

        for k, v in unk_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.5)

        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], '.', markerfacecolor='None', markeredgecolor='white', mew=0.6)

        # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)
        plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)

        # plots contours
        # for index in range(1, labelled_img.max()):
        #     label_i = regions[index].label
        #     contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
        #     y, x = contour.T
        #     plt.plot(x, y, color='cyan')

        for index in range(1, op.max()):
            label_i = regions[index].label
            contour = measure.find_contours(op == label_i, 0.8)[0]
            y, x = contour.T
            plt.plot(x, y, color='cyan')

        # plt.axis('off')
        # plt.savefig('Climp_series12_junc_representation_iso_fuz_unk_2_new_colors', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.close()

        plt.show()
        #markerfacecolor='None', markeredgecolor='pink', mew=0.4)
        #
        # if len(fuz) > 0:
        #     plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue', mew=0.6)
        #
        # # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red', mew=0.6)

        for k, v in iso_cc_coords.items():
            plt.plot(v[1], v[0], 's', markerfacecolor='magenta', markeredgecolor='magenta', mew=0.35, ms=20)

        for k, v in fuz_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='blue', mew=0.35)

        for k, v in unk_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.5)

        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], '.', markerfacecolor='None', markeredgecolor='white', mew=0.6)

        # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)
        plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.6)

        # plots contours
        # for index in range(1, labelled_img.max()):
        #     label_i = regions[index].label
        #     contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
        #     y, x = contour.T
        #     plt.plot(x, y, color='cyan')

        for index in range(1, op.max()):
            label_i = regions[index].label
            contour = measure.find_contours(op == label_i, 0.8)[0]
            y, x = contour.T
            plt.plot(x, y, color='cyan')

        # plt.axis('off')
        # plt.savefig('Climp_series12_junc_representation_iso_fuz_unk_2_new_colors', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.close()

        plt.show()

def crop_cc_from_saved():
    for i in range(100):
        img = imageio.imread(
            f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junction_crops/S13_j10_27_48_CC_crops/R13_decon_t0{i:02d}_ch01.png')
        x = 80
        y = 140
        img = img[:, :, :3]
        op = img[x - 8:x + 8, y - 8:y + 8]
        plt.axis('off')
        plt.imshow(op, cmap='gray', interpolation=None)
        plt.savefig(
            f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junction_crops/S13_j10_cc_area_v2/R13_decon_t0{i:02d}_ch01.png',
            bbox_inches='tight', pad_inches=0)
        plt.show()
        # plt.close()


# crop_cc_from_saved()
# exit()