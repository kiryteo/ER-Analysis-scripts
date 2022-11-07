

def get_voronoi(nps, skdata):

    vor = Voronoi(nps)

    # fig, ax = plt.subplots()
    fig = voronoi_plot_2d(vor)
    # plt.figure(num=1, figsize=(8,8))

    # plt.title('Voronoi for ATL Series 10, Blue points: Reference junctions, Red points: per frame junctions')
    # img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/er_mean/%s_er_mean.png'%(f'{group}', f'{group.lower()}{num_series}'))

    plt.plot(nps[:, 0], nps[:, 1], 'o', markerfacecolor='None', markeredgecolor='blue', mew=3)
    plt.plot(skdata[:, 0], skdata[:, 1], 'o', markerfacecolor='None', markeredgecolor='red')
    # plt.imshow()
    # plt.gca().set_aspect(1)
    # plt.axis('scaled')

    # plt.savefig('Voro_ATL_10_new.png', bbox_inches='tight')

    f = plt.gcf()
    f.set_size_inches(8, 8)

    plt.show()
