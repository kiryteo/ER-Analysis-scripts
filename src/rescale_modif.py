def plot_junc_areas(group, series_num, labelled_img, iso, fuz, skdata, iso_cc_coords, fuz_cc_coords, unk_cc_coords):


    # plt.imshow(labelled_img)
    # plt.show()

    bin_lab =

    dil_lab = skimage.morphology.binary_dilation(labelled_img)
    op = dil_lab - labelled_img

    op = skimage.transform.rescale(op, 8, order=0, anti_aliasing=False).astype('float64')

    # plt.imshow(op)
    # plt.show()
    #
    # exit()

    # regions = regionprops(labelled_img)
    regions = regionprops(op)
    # print(op.dtype)
    # print(labelled_img.dtype)

    # print(regions)
    # exit()

    for i in range(1):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(confocal_data_path + f'{group}/files/img_{series_num}_decon_t0{i:02d}.tif')

        else:
            img = imageio.imread(confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

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
        #     plt.plot(v[1], v[0], '.', markerfacecolor='None', markeredgecolor='pink', mew=0.4)
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