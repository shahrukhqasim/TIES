import numpy as np
import cv2
import math


class NeighborGraphBuilder:
    def __init__(self, all_tokens_rects, image):
        self.all_tokens_rects = all_tokens_rects
        self.image = image

    # if (max(0, std::min(i.x + i.width, j.x + j.width) - std::max(i.x, j.x)) > 0)
    @staticmethod
    def horizontal_overlap(A, B):
        if max(0, min(A['x'] + A['width'], B['x'] + B['width']) - max (A['x'], B['x'])) > 0:
            return True
        return False

    @staticmethod
    def vertical_overlap(A, B):
        if max(0, min(A['y'] + A['height'], B['y'] + B['height']) - max (A['y'], B['y'])) > 0:
            return True
        return False

    def get_neighbor_matrix(self):
        N = len(self.all_tokens_rects)
        m = np.ones((N, 4)) * (-1)
        n = np.ones((N, 4)) * (-1)
        height, width = np.shape(self.image)
        for i in range(N):
            A = self.all_tokens_rects[i]
            min_left = min_right = width
            min_top = min_bottom = height
            min_index_top = min_index_bottom = min_index_left = min_index_right = -1
            for j in range(N):
                if i == j:
                    continue
                B = self.all_tokens_rects[j]
                if self.horizontal_overlap(A, B):
                    new_distance = A['y'] - B['y']
                    new_distance_abs = abs(new_distance)
                    # B is above A
                    if new_distance > 0 and new_distance < min_top:
                        min_top = new_distance
                        min_index_top = j
                    # B is below A
                    elif new_distance < min_bottom:
                        min_bottom = new_distance
                        min_index_bottom = j
                if self.vertical_overlap(A, B):
                    new_distance = A['x'] - B['x']
                    new_distance_abs = abs(new_distance)
                    # B is left of A
                    if new_distance > 0 and new_distance < min_left:
                        min_left = new_distance
                        min_index_left = j
                    # B is below A
                    elif new_distance < min_right:
                        min_right = new_distance
                        min_index_right = j
            m[i,0] = min_index_left
            m[i,1] = min_index_top
            m[i,2] = min_index_right
            m[i,3] = min_index_bottom

            n[i, 0] = min_left
            n[i, 1] = min_top
            n[i, 2] = min_right
            n[i, 3] = min_bottom
        return m, n