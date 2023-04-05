import networkx as nx
import numpy as np
import unittest

from junction_analysis_modules import JunctionAnalysis as JA

class TestGetJunctions(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(1, {'o': np.array([0, 0])}),
                                   (2, {'o': np.array([1, 0])}),
                                   (3, {'o': np.array([1, 1])}),
                                   (4, {'o': np.array([0, 1])})])
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

    def test_get_junctions(self):
        junctions = JA.get_junctions(self.graph)
        expected_junctions = [np.array([1, 0.]), np.array([0, 1.])]
        self.assertListEqual(junctions, expected_junctions)
