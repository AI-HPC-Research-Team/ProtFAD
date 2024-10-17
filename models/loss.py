import torch
from torch import nn
import torch.nn.functional as F

def contrastive_loss(emb1, emb2, temp):
    with torch.no_grad():
        temp.clamp_(0.001,0.5)

    emb1 = F.normalize(emb1, dim=-1, p=2)
    emb2 = F.normalize(emb2, dim=-1, p=2)
    # cosine similarity as logits
    sim_x2y = emb1 @ emb2.t().detach() / temp
    sim_y2x = emb2 @ emb1.t().detach() / temp
    loss_x2y = -torch.sum(F.log_softmax(sim_x2y, dim=1), dim=1).mean()
    loss_y2x = -torch.sum(F.log_softmax(sim_y2x, dim=1), dim=1).mean()
    return (loss_x2y + loss_y2x) / 2

class ContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss
    from: https://blog.csdn.net/weixin_44966641/article/details/120382198
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))			# 超参数 温度
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        batch_size = emb_i.shape[0]
        device = emb_i.device
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float()

        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
