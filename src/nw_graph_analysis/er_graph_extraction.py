# Pipeline to get the ER graph starting with skeleton


class ERGraphExtraction:

    def __init__(self, confocal_data_path):
        self.confocal_data_path = confocal_data_path

    def get_vess_enh_er(self):
        """
        Get the vessel enhanced input for skeletonization
        """
        pass

    def get_skel(self):
        """
        Get skeleton from vessel enhanced input
        """
        pass

    def get_graph_from_skel(self):
        """
        Get a networkx graph from skeleton input
        """
        pass