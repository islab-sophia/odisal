import torch


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def composite_avg(x):  # x (1, 3, 120, 120)
    out = torch.zeros(1, 1, 120, 120).to(DEVICE)
    out[0] = torch.mean(input=x, dim=1)
    return out


def composite_max(x):
    out = torch.zeros(1, 1, 120, 120).to(DEVICE)
    out[0] = torch.max(input=x, dim=1).values
    return out