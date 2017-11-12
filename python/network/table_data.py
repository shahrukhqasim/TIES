
class TableData:
    def __init__(self, tokens_embeddings, tokens_rects, neighbor_distance_matrix, tokens_neighbor_matrix,
                 tokens_share_col_matrix, tokens_share_row_matrix, tokens_share_cell_matrix):
        self.embeddings = tokens_embeddings
        self.rects = tokens_rects
        self.distances = neighbor_distance_matrix
        self.neighbor_graph = tokens_neighbor_matrix
        self.row_share = tokens_share_row_matrix
        self.col_share = tokens_share_col_matrix
        self.cell_share = tokens_share_cell_matrix

