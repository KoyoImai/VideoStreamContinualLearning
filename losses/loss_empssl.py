import torch.nn as nn
import torch
import torch.nn.functional as F


class Similarity_Loss(nn.Module):

    def __init__(self):

        super().__init__()
        pass

    def forward(self, z_list, z_avg):

        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out


class TotalCodingRate(nn.Module):

    def __init__(self, eps=0.01):

        super(TotalCodingRate, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        p, m = W.shape

        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.


    def forward(self, X):

        return - self.compute_discrimn_loss(X.T)