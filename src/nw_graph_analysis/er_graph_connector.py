# add all the modules for connecting a graph based on near node connectivity



def node_connector(path_er, path_frame):

    graph = skel_to_graph(path_frame)

    # all junction from the graph with degree > 2
    junc_analysis = JA(confocal_data_path)
    junctions = junc_analysis.get_junctions(graph)

    relevant_nodes = np.array(junctions)

    fin_dict, g_nodes_array = get_updated_neighbor_dict(graph)

    temp_graph = copy.deepcopy(graph)

    # er_proc = imageio.imread(path_er_proc)
    # er_proc_bg = np.where(er_proc==0)

    er_input = imageio.imread(path_er)
    cost_arr = np.ones((128, 128))
    # cost_arr[er_proc_bg] = 0

    for node in dict(graph.degree()):
        # access the first element of graph.neighbors
        neighbor = next(iter(graph.neighbors(node)))

        connect_nodes(er_input, temp_graph, node, neighbor, fin_dict, cost_arr, g_nodes_array)

    tgraph = copy.deepcopy(temp_graph)
    for node in temp_graph.nodes():
        process_node(tgraph, node)

    tgraph2 = copy.deepcopy(tgraph)
    for node in tgraph.nodes():
        process_node(tgraph2, node)

    # deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(tgraph2)

    return tgraph2