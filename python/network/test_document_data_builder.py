import unittest
from network.neighbor_graph_builder import NeighborGraphBuilder
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_matrix(self):
        builder = NeighborGraphBuilder([{'x' : 500, 'width': 50, 'y' : 500, 'height' : 100} , {'x' : 300, 'width': 100, 'y' : 500, 'height' : 100},
                                        {'x': 500, 'width': 100, 'y': 300, 'height': 100}, {'x' : 500, 'width': 100, 'y' : 700, 'height' : 100}], np.zeros((1000,1000)))
        m = builder.get_neighbor_matrix()
        assert(m[0,0] == 1)
        assert(m[0,1] == 2)
        assert(m[0,3] == 3)
        assert(m[0,2] == -1)
        assert(m[1,0] == -1)
        assert(m[1,1] == -1)
        assert(m[1,2] == 0)
        assert(m[1,3] == -1)

    def test_matrix_2(self):
        builder = NeighborGraphBuilder([{'x' : 500, 'width': 50, 'y' : 500, 'height' : 100} , {'x' : 300, 'width': 100, 'y' : 500, 'height' : 100},
                                        {'x': 500, 'width': 100, 'y': 300, 'height': 100}, {'x' : 500, 'width': 100, 'y' : 700, 'height' : 100}], np.zeros((1000,1000)))
        m = builder.get_neighbor_matrix()
        assert(m[0,0] == 1)
        assert(m[0,1] == 2)
        assert(m[0,3] == 3)


if __name__ == '__main__':
    unittest.main()
