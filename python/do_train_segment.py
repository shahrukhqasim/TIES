from network import table_segment_network as ts

network = ts.TableSegmentNetwork()
network.construct_graphs()
network.train()