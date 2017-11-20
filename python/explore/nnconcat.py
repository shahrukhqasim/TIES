import torch


def expand_way_1(A):
    return A.view(1,10,2).expand(10,1,10,2).transpose(0,2).contiguous().view(100,2)
def expand_way_2(A):
    return A.expand(10,10,2).contiguous().view(100,2)


A = torch.cat((torch.unsqueeze(torch.range(1,10), dim=1), torch.unsqueeze(2*torch.range(1,10), dim=1)), dim=1)
print(A)
A1=expand_way_1(A)
A2=expand_way_2(A)
print(A1)
print(A2)

A3=torch.cat((A1,A2), dim=1)

print(A3)