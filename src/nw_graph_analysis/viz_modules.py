import imageio
import sknw
import matplotlib.pyplot as plt
import numpy as np
import copy
from PIL import Image


def get_graph(skel):
    return sknw.build_sknw(skel, iso=False, multi=True)


def create_junc_crop_sequence():
    l = []
    for i in range(100):
        img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/junction_crops/S13_j10_cc_area_v2/R13_decon_t0{i:02d}_ch01.png')
        img = img[:,:,:3]
        l.append(img)

    def seq_with_plt():
        img = np.concatenate(l, axis=1)

        print(img.shape)

        fig = plt.gca()

        plt.axis('off')
        plt.imshow(img, interpolation=None)

        fig = plt.gcf()
        fig.set_size_inches(24, 246, forward=True)

        # fig.savefig('R13_j10_mch.tif', bbox_inches='tight', pad_inches=0)
        plt.show()

    def seq_with_pil():
        images = list(map(Image.open, l))
        # images = list(map(Image.fromarray, l))
        w, h = zip(*(i.size for i in images))

        tw = sum(w)
        mxh = max(h)

        new_im = Image.new('RGB', (tw, mxh))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save('A1_j15_sequence_mCherry.png')


def plot_nodes_on_er_input(attribute):
    # input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean/climp12_er_mean.png')
    input = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0.png')

    skel = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_enhance_skel.png')

    graph = sknw.build_sknw(skel, iso=False)
    plt.axis('off')

    if attribute == 'nodes':
        plt.imshow(input, cmap='gray')
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])

        newps = [ps[j] for j, val in enumerate(graph.degree) if val[1] > 2]
        nps = [[each[0], each[1]] for each in newps]

        plt.plot(nps[:, 1], nps[:, 0], 'o', markerfacecolor='red', markeredgecolor='red', mew=0.5, markersize=4)

        plt.show()
        # plt.savefig('nodes_C12_t0_input.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
    elif attribute == 'edges':
        # draw edges with start and end nodes
        plt.imshow(input, cmap='gray')
        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:,1], ps[:,0], 'green')

            nodes = graph.nodes()
            ps = np.array([nodes[i]['o'] for i in nodes])
            # plt.plot(ps[:,1], ps[:,0], 'r.')
            # plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='red', markeredgecolor='red', mew=0.5, markersize=4)

            # plt.savefig('edge_nodes_C12_t0.png', bbox_inches='tight', pad_inches=0)
            # plt.close()
            plt.show()
    elif attribute == 'relevant_edges':
        node_set = graph.nodes()
        edge_set = graph.edges()

        s = copy.deepcopy(skel)
        s[np.where(s>0)] = 0

        # draw image
        plt.axis('off')
        plt.imshow(s, cmap='gray')
        # plt.imshow(imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png'), cmap='gray')


        for (s,e) in graph.edges():
            ps_og = graph[s][e][0]['pts']
            try:
                ps_multi = graph[s][e][1]['pts']
                plt.plot(ps_multi[:,1], ps_multi[:,0], 'cyan')
            except Exception:
                pass
            plt.plot(ps_og[:,1], ps_og[:,0], 'cyan')

        # plt.savefig('A1_multigraph_sections_skel.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        # # plt.show()
        #
        # exit()
        #
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        # # plt.plot(ps[:,1], ps[:,0], 'r.')
        plt.plot(ps[:, 1], ps[:, 0], 'x', markerfacecolor='yellow', markeredgecolor='yellow', mew=0.5, markersize=3)


        newps = [ps[j] for j, val in enumerate(graph.degree) if val[1] > 2]

        relevant_node_list = []
        for npt in newps:
            for each in node_set:
                node_val = node_set[each]['o']
                # print(node_set[each]['o'])
                if npt[0] == node_val[0] and npt[1] == node_val[1]:
                    relevant_node_list.append(each)

        nps = [[each[0], each[1]] for each in newps]
        relevant_edge_terminals = []
        relevant_edge_list = []
        for (start_node, end_node) in edge_set:
            if start_node in relevant_node_list and end_node in relevant_node_list:
                tubule_coords = graph[start_node][end_node][0]['pts']
                relevant_edge_terminals.append((start_node, end_node))
                relevant_edge_list.append(tubule_coords)
                # plt.plot(tubule_coords[:,1], tubule_coords[:,0], 'green')

        for each in relevant_edge_list:
            plt.plot(each[:, 1], each[:, 0], 'green')

        nps = np.array(nps)
        plt.plot(nps[:, 1], nps[:, 0], 'o', markerfacecolor='red', markeredgecolor='red', markersize=3)
        #
        # # title and show
        # # plt.title('Build Graph')
        # plt.savefig(pre + 'A1_decon_t0%s_ch00_graph.png'%f'{i:02d}', bbox_inches='tight')
        # plt.savefig('A1_relevant_node_graph.png', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.savefig('A1_er_nodes.png', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.savefig('climp12_mean_proj_junctions_only.png', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.show()


# def plot_junc_areas(group, series_num, iso, fuz, unk, labelled_img):
def plot_junc_areas(group, series_num, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords, labelled_img):
    regions = regionprops(labelled_img)

    bin_reg = (labelled_img > 0)
    contours = measure.find_contours(bin_reg)


    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=0.8, color='cyan')

    # plt.show()

    # plt.imshow(labelled_img)
    # plt.show()

    # exit()

    for i in range(1):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(confocal_data_path + f'{group}/files/img_{series_num}_decon_t0{i:02d}.tif')

        else:
            img = imageio.imread(confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        # plt.plot(iso[:, 1], iso[:, 0], 's', markerfacecolor='None', markeredgecolor='red')

        # plt.plot(skdata[:, 1], skdata[:, 0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        # for k, v in iso_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.4)
        #
        # for k, v in fuz_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.4)
        #
        # for k, v in unk_cc_coords.items():
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='pink', mew=0.4)
        #
        # if len(fuz) > 0:
        #     plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue', mew=0.6)
        #
        # # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        # plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red', mew=0.6)

        for k, v in iso_cc_coords.items():
            # plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='magenta', mew=0.35)
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='red', mew=0.15)#0.35)

        for k, v in fuz_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='blue', mew=0.15)#0.35)

        for k, v in unk_cc_coords.items():
            plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='green', mew=0.25)#0.5)

        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], '.', markerfacecolor='None', markeredgecolor='white', mew=0.3)#0.6)

        # plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        plt.plot(iso[:, 1], iso[:, 0], '.', markerfacecolor='None', markeredgecolor='yellow', mew=0.3)#0.6)



        # plots contours
        # for index in range(1, labelled_img.max()):
        #     label_i = regions[index].label
        # contour = measure.find_contours(labelled_img == label_i, 0.8)[0]
        # y, x = contour.T
        # plt.plot(x, y, color='cyan')

        plt.axis('off')
        plt.savefig(f'{group}_series{series_num}_junc_representation_iso_fuz_unk_2_new_colors', bbox_inches='tight', pad_inches=0, dpi=700)
        plt.close()
        #
        # plt.show()


def fuz_isolated_junctions(group, series_num):

    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.

    # plt.imshow(spread_img)
    # plt.show()
    #
    # exit()

    labelled_img = label(spread_img, connectivity=2)

    # plt.imshow(labelled_img)
    # plt.show()
    #
    # exit()

    regions = regionprops(labelled_img)

    # cc_list = []
    # for idx in range(1, labelled_img.max()):
    #     lab_i = props[idx].label

    cc_area_dict = {idx: props.area for idx, props in enumerate(regions)}
    # print(cc_area_dict)

    # exit()

    num_components = np.unique(labelled_img)

    label_vals, assigned_components = get_junction_types(nps, labelled_img)

    unassigned_cc_dict = get_uncertain_junctions(labelled_img, skdata, num_components, assigned_components)

    isolated_junc = []
    isolated_junc_area = []
    fuzzy_junc = []
    fuzzy_junc_area = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
                isolated_junc_area.append(cc_area_dict[k])
            else:
                fuzzy_junc.append(v)
                fuzzy_junc_area.append(cc_area_dict[k])

    print(isolated_junc_area)
    print(fuzzy_junc_area)

    unknown_junc = [v for k, v in unassigned_cc_dict.items()]
    iso = np.array(isolated_junc)

    fuz = list(itertools.chain.from_iterable(fuzzy_junc))
    fuz = np.array(fuz)

    unk = list(itertools.chain.from_iterable(unknown_junc))
    unk = np.array(unk)

    # img = imageio.imread(
    #     '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')

    for i in range(100):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(confocal_data_path + f'{group}/files/img_{series_num}_decon_t0{i:02d}.tif')

        else:
            img = imageio.imread(confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red')
        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue')
        plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        plt.plot(skdata[:, 1], skdata[:, 0], 'x', markerfacecolor='None', markeredgecolor='yellow')
        plt.show()
        # if group == 'Control':
        #     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/Ct%s_decon_t0%s_ch00.png'%(f'{group}', f'{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        # else:
        #     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        # plt.close()


fuz_isolated_junctions('Control', 13)

# for i in range(1, 27):
#     fuz_isolated_junctions('ATL', i)

exit()

def plot_junc_spread(group, n1, n2, num_series):
    fig = plt.gcf()
    ax = fig.gca()
    # gr = cm.Greens(np.linspace(n3arr.min()[0], n3arr.max()[0], num=len(n3)))
    # mcmap = mcolors.LinearSegmentedColormap.from_list('mcmap', gr)
    img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/er_mean/{group.lower()}{num_series}_er_mean.png'))

    plt.imshow(img, cmap='gray', interpolation='none')
    # plt.plot(n2[:, 1], n2[:, 0], 'b.')
    plt.scatter(n2[:, 1], n2[:, 0], c=n3val, cmap='Blues', marker='o')
    # plt.colorbar()
    plt.plot(n1[:, 1], n1[:, 0], 'o', markerfacecolor='None', markeredgecolor='red', mew=1.5)  # , ms=4)
    # plt.plot(n1[:, 1], n1[:, 0], 'r.')
    # c = Circle((n1[0, 1], n1[0, 0]), radius=3, linewidth=2, facecolor='none', edgecolor='green', alpha=0.7)
    # ax.add_patch(c)
    # plt.plot(n2[:, 1], n2[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue')

    for i, each in enumerate(n1):
        c1 = plt.Circle((n1[i, 1], n1[i, 0]), 3, color='r', fill=False, linestyle='--')
        ax.add_patch(c1)
        # if n5[i] > med:
        s = '(' + '%.2f' % n5[i] + ',' + str(n6[i]) + ')'
        # s = '(' + str(n1[i,1]) + ',' + str(n1[i,0]) + ',' + '%.2f'%n5[i] + ',' + str(n6[i]) + ')'
        ax.text(n1[i, 1], n1[i, 0], s, c='yellow')

    plt.axis('off')
    plt.suptitle(f'Climp series {num_series} junctions movement variance')
    plt.title('Variance of list with distances for matched junctions per frame w.r.t. reference frame junctions')
    # plt.savefig('RTN1_junc_spread.png', bbox_inches='tight', pad_inches=0)
    plt.show()


plot_junc_spread('Climp', n1, n2, 3)
exit()


def junction_location_plotter(group, num_series):
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'


    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)

    # print(nps[46, 1], nps[46, 0])
    #
    # exit()
    # plt.imshow(mean_proj_img, cmap='gray')

    # for (s,e) in g.edges():
    #     ps = g[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'green')
    # for dr in range(len(nps)):
    #     os.makedirs('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/junc%s'%f'{dr+1}')

    for frame in range(100):

        # ER Input image
        if group == 'Control':
            path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

        else:
            path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        # ER Skel image
        # sk_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}'
        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'


        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        sk_nps = np.array(sk_nps)

        # img = img * 255.
        # # plt.plot(ps[:, 1], ps[:, 0], 'y.')
        # for j in range(47, 48):
        # plt.figure(figsize=(128/77, 128/77))
        plt.imshow(img, cmap='gray')
        # plt.imshow(imageio.imread(img), cmap='gray')
        plt.axis('off')
        # plt.title('t=%s'%f'{i}')
        plt.plot(nps[:, 1], nps[:, 0], 'r.')
        plt.plot(sk_nps[:, 1], sk_nps[:, 0], 'b.')
        # y = nps[0, 1]
        # x = nps[0, 0]
        # cv2.rectangle(img, (x-1, y-1), (x+1, y+1), (0, 0, 255), 2)

        plt.savefig((confocal_data_path + f'{group}/new_op_jul/{group.lower()}_junc_viz/{group_pref[group]}{num_series}_t{frame:02d}.png'), bbox_inches='tight', pad_inches=0)


        plt.close()


# junction_location_plotter('RTN', 1)
# exit()

nps, skdata, labelled_img = label_junctions('Climp', 12)
# exit()

label_vals, unassigned_cc_dict = separate_junc_cc(nps, skdata, labelled_img)
iso, fuz, unk = get_junction_areas(label_vals, unassigned_cc_dict)

# mp_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png')
mp_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean/climp12_er_mean.png')

plt.imshow(mp_frame, cmap='gray')


# plt.plot(iso[:, 1], iso[:, 0], 'x', markerfacecolor='None', markeredgecolor='red', mew=0.6)

# exit()

g = sknw.build_sknw(mp_frame, iso=False)
G = nx.Graph()

G.add_nodes_from(g.nodes)
G.add_edges_from(g.edges)

ps = np.array([g.nodes[i]['o'] for i in g.nodes])
# print(ps)

# plt.plot(ps[:, 1], ps[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue', mew=0.6)
#
# plt.show()
# exit()


iso_ps_ids = [i for i, val in enumerate(G.degree) if val[1] > 2]

for each in iso_ps_ids:
    if g.nodes[each]['o'] in fuz:
        G.remove_node(each)
    # print(g.nodes[each]['o'])


gl = G.nodes
iso_ps = np.array([g.nodes[i]['o'] for i in gl])
# print(G.edges)
# G.remove_node()
# print(iso_ps)

plt.plot(iso_ps[:, 1], iso_ps[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue', mew=0.6)
plt.show()

exit()