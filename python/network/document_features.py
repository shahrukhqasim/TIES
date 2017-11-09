
class DocumentFeatures:
    def __init__(self, tokens_embeddings, tokens_rects, neighbor_distance_matrix, tokens_neighbor_matrix, tokens_classes):
        self.embeddings = tokens_embeddings
        self.rects = tokens_rects
        self.distances = neighbor_distance_matrix
        self.neighbor_graph = tokens_neighbor_matrix
        self.classes = tokens_classes

