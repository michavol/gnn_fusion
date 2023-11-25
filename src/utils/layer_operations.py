import torch
import torch.nn.functional as F


# (Weighted) Averaging of weight matrices of a layer
def get_avg_parameters(parameters, weights=None):
    #avg_pars = []
    #print(parameters.shape)
    if weights is not None:
        weighted_par_group = [par * weights[i] for i, par in enumerate(parameters)]
        avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
    else:
        # print("shape of stacked params is ", torch.stack(par_group).shape) # (2, 400, 784)
        avg_par = torch.mean(torch.stack(parameters), dim=0)
    print(type(avg_par))
    #avg_pars.append(avg_par)
    return avg_par