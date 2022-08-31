import skelnw
from plantcv import plantcv as pcv
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize
import cv2
from skimage import draw
import sknw

# img = np.ones((4,4))
# bf = np.pad(img, (1,1), mode='constant').astype(np.uint16)
#
# plt.imshow(img)
# plt.show()

# ske = skeletonize(~img).astype(np.uint16)

lt = []
for i in range(100):
    # img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A4_decon_t0%s_ch00.tif'%f'{i:02d}')
    #
    # img = (img - img.min()) / (img.max() - img.min())

    ske = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A4/A4_decon_t0%s_ch00_skel.png'%f'{i:02d}')

    graph = sknw.build_sknw(ske, iso=False)

    # plt.imshow(img, cmap='gray')

    # draw edges by pts
    # for (s,e) in graph.edges():
    #     ps = graph[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'green')

    # draw node by o
    nodes = graph.nodes()
    # ps = np.array([nodes[i]['o'] for i in nodes])
    # plt.plot(ps[:,1], ps[:,0], 'r.')
    lt.append(len(nodes))
    # title and show
    # plt.title('Skeleton to Graph')
    # plt.show()
    # plt.axis('off')
    # plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/graphs/A4/A4_decon_t0%s_ch00_graph.png'%f'{i:02d}', bbox_inches='tight', pad_inches=0)
    # plt.close()

plt.plot(lt)
plt.title('')
# plt.hist(lt)
plt.show()

exit()

for i in range(100):
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A4_decon_t0%s_ch00.tif'%f'{i:02d}')

    img = (img - img.min()) / (img.max() - img.min())

    ske = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A4/A4_decon_t0%s_ch00_skel.png'%f'{i:02d}')

    graph = sknw.build_sknw(ske, iso=False)

    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:,1], ps[:,0], 'r.')

    # title and show
    # plt.title('Skeleton to Graph')
    # plt.show()
    plt.axis('off')
    plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/graphs/A4/A4_decon_t0%s_ch00_graph.png'%f'{i:02d}', bbox_inches='tight', pad_inches=0)
    plt.close()

exit()

for i in range(1):
    ske = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/skel/C29/C29_decon_t0%s_ch00_skel.png'%f'{i:02d}')

    # ske = ske[::-1]

    # build graph from skeleton
    graph = sknw.build_sknw(ske, iso=False)

    # draw image
    # plt.rcParams['figure.figsize'] = 128, 128

    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(1,1)
    #
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    # ax.imshow(ske, cmap='gray', aspect='auto')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
    plt.margins(0, 0)

    # draw edges by pts
    # for (s,e) in graph.edges():
    #     ps = graph[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'yellow')

    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])

    overlay_image = cv2.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C29_decon_t0%s_ch01_std.png'%f'{i:02d}')
    # overlay_image = (overlay_image - overlay_image.min()) / (overlay_image.max() - overlay_image.min())

    lab = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2LAB)
    l_c, a, b = cv2.split(lab)


    # overlay_image = overlay_image.astype(np.float32)
    #
    clahe = cv2.createCLAHE(clipLimit=15.0)
    cl1 = clahe.apply(l_c)

    limg = cv2.merge((cl1, a, b))

    enh_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


    for (s, e) in graph.edges():
        pse = graph[s][e]['pts']
        rl, cl = draw.line(pse[0][0], pse[0][1], pse[-1][0], pse[-1][1])
        for a, b in zip(rl, cl):
            enh_img[a, b] = [255, 255, 0]

    for pts in ps:
        rr, cc = draw.circle_perimeter(pts[0], pts[1], radius=1, shape=(128, 128))
        for a, b in zip(rr, cc):
            enh_img[a, b] = [255, 0, 0]

    # plt.imshow(overlay_image)
    plt.imshow(enh_img)
    plt.show()
    # plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/Climp/graph_overlay_endpts/C29_decon_t0%s_ch01_graph_overlay.png'%f'{i:02d}', bbox_inches='tight', pad_inches=0)
    # plt.close()
# plt.plot(ps[:,1], ps[:,0], 'r.')


# title and show
# plt.title('Build Graph')
# plt.show()

# plt.imsave('new_grppppp.png', )
# plt.tight_layout()
# plt.savefig('new_grp.png')
# plt.savefig('new_grphhh.png', bbox_inches='tight', pad_inches=0)
# plt.close()
# fig.savefig('f.png', 128)

exit()




def get_edges():
    for i in range(1):
        pre = '/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/'

        # open and skeletonize
        ske = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}')

        # ske = np.flipud(ske)
        ske = ske[::-1]

        # build graph from skeleton
        # graph = skelnw.build_sknw(ske)
        nodes, edges = skelnw.build_sknw(ske)

        for i, node in enumerate(nodes):
            if i == 6:
                break
            print(node)
        # print(edges[0])

        #for node in nodes:

        # draw image
        plt.axis('off')
        # plt.imshow(ske, cmap='gray')

        # draw edges by pts
        # for (s,e) in graph.edges():
        #     ps = graph[s][e]['pts']
        #     plt.plot(ps[:,1], ps[:,0], 'green')
        #
        # # draw node by o
        # nodes = graph.nodes()
        # ps = np.array([nodes[i]['o'] for i in nodes])
        # plt.plot(ps[:,1], ps[:,0], 'r.')
        #
        # # title and show
        # # plt.title('Build Graph')
        # # plt.savefig(pre + 'A1_decon_t0%s_ch00_graph.png'%f'{i:02d}', bbox_inches='tight')
        # plt.show()
        # plt.close()


def res_change():
    grph = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/A1_decon_t000_ch00_graph.png')

    skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1/A1_decon_t000_ch00_skel.png')

    er = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/A1_decon_t000_ch01_std_std_adj.png')

    grph_pil = Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1-graph/A1_decon_t000_ch00_graph.png')

    er_pil = Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/A1_decon_t000_ch01_std_std_adj.png')

    grph_pil = np.asarray(grph_pil)
    grph_pil = grph_pil[:,:,:3]

    print(grph_pil.dtype)

    er_pil = np.asarray(er_pil)
    print(er_pil.dtype)
    er_pil = resize(er_pil, (389, 515)).astype('uint8')
    print(er_pil.dtype)
    er_pil = np.stack((er_pil, er_pil, er_pil), axis=2)

    print(er_pil.shape)
    print(grph_pil.shape)

    op = 0.5 * er_pil + 0.5 * grph_pil

    # grph_pil = Image.fromarray(grph_pil)
    # # print(grph_pil.dtype)
    # er_pil = Image.fromarray(er_pil)
    #
    # op = Image.blend(er_pil, grph_pil, 0.5)
    plt.imshow(op)
    plt.show()

    #
    # print(grph.shape)
    # print(grph.size)
    #
    # grph = grph[:,:,:3]
    #
    # print(grph.shape)
    # print(grph.size)
    #
    # # fig = plt.figure(figsize=(9,4))
    # # r, c = 1, 2
    # #
    # # fig.add_subplot(r, c, 1)
    # # plt.imshow(grph)
    #
    # grph = grph[4:385, 4:511]
    #
    # # skel = resize(skel, (381, 507))
    # #
    # # skel = np.stack((skel, skel, skel), axis=2)
    #
    # er = resize(er, (381, 507))
    #
    # er = np.stack((er, er, er), axis=2)
    #
    # op = 0.5 * er + 0.5 * grph
    # plt.imshow(op, cmap='gray_r')
    # plt.show()

    # fig.add_subplot(r, c, 2)
    # plt.imshow(grph)

    # grph = resize(grph, (389, 515))

    # plt.imshow(grph, cmap='gray_r')
    # plt.show()

    # imageio.imwrite('grph_res.png', grph)

res_change()
exit()

def pad_res_change():
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1/A1_decon_t000_ch00_skel.png')

     # print(buf.shape)
    res = resize(img, (387, 513))

    buf = np.pad(res, (1,1), mode='constant')

    imageio.imwrite('rerere.png', buf)

    # plt.axis('off')
    # plt.imshow(res)
    # # plt.show()
    # plt.savefig('res_skel.png', bbox_inches='tight')
    # plt.close()



# for i in range(100):
#     skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}')
#     skel = np.flipud(skel)
#     graph = sknw.build_sknw(skel, iso=False)
#
#     plt.axis('off')
#     plt.imshow(skel, cmap='gray')
#
#     # draw edges by pts
#     for (s,e) in graph.edges():
#         ps = graph[s][e]['pts']
#         plt.plot(ps[:,1], ps[:,0], 'green')
#
#     # draw node by o
#     nodes = graph.nodes()
#     ps = np.array([nodes[i]['o'] for i in nodes])
#     plt.plot(ps[:,1], ps[:,0], 'r.')
#
#     # title and show
#     plt.title('A1_decon_t0%s_ch00'%f'{i:02d}')
#     plt.savefig('A1_decon_t0%s_ch00_graph.png'%f'{i:02d}', bbox_inches='tight')
#     plt.close()
#     # plt.show()
#
#
