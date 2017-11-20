
class TableDetectDocument:
    def __init__(self, tokens_embeddings, tokens_rects, neighbor_distance_matrix, tokens_neighbor_matrix, tokens_classes, conv_features, inside_same_table):
        self.embeddings = tokens_embeddings
        self.rects = tokens_rects
        self.distances = neighbor_distance_matrix
        self.neighbor_graph = tokens_neighbor_matrix
        self.classes = tokens_classes
        self.conv_features = conv_features
        self.inside_same_table = inside_same_table