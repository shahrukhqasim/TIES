import numpy as np


def union(r1, r2):
    left = min(r1[0], r2[0])
    right = max(r1[0] + r1[2], r2[0] + r2[2])
    top = min(r1[1], r2[1])
    bottom = max(r1[1] + r1[3], r2[1] + r2[3])

    return np.array([left, top, right - left, bottom - top])


if __name__ == '__main__':
    r1 = [50,50,100,100]
    r2 = [100,100,100,100]

    print(union(r1,r2))