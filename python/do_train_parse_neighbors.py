import os
import sys
from network import data_features_dumper as dr
from network import computation_graph
from network.trainer_neighbor_parse import Trainer


# path = '/home/srq/Datasets/tables/unlv-for-nlp/train'
# glove_path = '/media/srq/Seagate Expansion Drive/Models/GloVe/glove.840B.300d.txt'
#
# data_reader = dr.DataReader(path, glove_path, 'train')
# data_reader.load()


trainer = Trainer()
trainer.init(dump_features_again=False)
trainer.train()