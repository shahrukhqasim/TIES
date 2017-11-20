import numpy as np
import torch


def sample_indices(vector):
    A = vector.copy()
    num_ones = np.sum(A == 1)
    print(num_ones)

    sample_from = np.where(A==0)[0]
    np.random.shuffle(sample_from)
    picked_zeros = sample_from[0:min(num_ones, np.size(sample_from))]
    A[picked_zeros] = 1

    return np.where(A == 1)[0]




total_num = 0.0
count = 0

for i in range(10000):
    A = np.round(np.random.rand(30))
    #
    # A[np.random.randint(0,np.size(A), size=10)]=1

    total_num += np.size(sample_indices(A))
    count+=1

print("Average number =", total_num/count)

0/0

ones = np.sum(A==1)

for i in range(10000):
    B = A.copy()
    B[np.random.randint(0, np.size(A), size=ones)] = 2
    print(np.sum(B==1)+np.sum(B==2))


