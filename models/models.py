import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import random
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import fps, global_max_pool, global_mean_pool, radius

from .cdconv import StructEncoder
from .projector import *
from .modules import Linear, MLP, DomainAttention, DomainMeaning

#define triplet loss function
class triplet_loss(nn.Module): 
    def __init__(self, margin=0.2): 
        super(triplet_loss, self).__init__() 
        self.margin = margin
    def forward(self, anchor, positive, negative): 
        pos_dist = (anchor - positive).pow(2).sum(1) 
        neg_dist = (anchor - negative).pow(2).sum(1) 
        loss = F.relu(pos_dist - neg_dist + self.margin) 
        return loss.mean()

def cosine_distance(x, y):
    return 1 - F.cosine_similarity(x, y, dim=-1)

class Model(nn.Module):
    def __init__(self,
                 domain_dim: int = 256,
                 max_domain_num: int = 16,
                 seq_emb_dim: int = 0,
                 cdconv_param: dict = None,
                 feat_dim: int = 768,
                 fad_emb: bool = True,
                 domain_num: int = 0,
                 domain_attention: bool = True,
                 protein_cropping: bool = True,
                 contrastive_loss: bool = False,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        self.seq_emb_dim = seq_emb_dim
        self.domain_dim = domain_dim
        self.contrastive_loss = contrastive_loss
        self.protein_cropping = protein_cropping
        self.fad_emb = fad_emb

        if cdconv_param is not None:
            struct_dim = cdconv_param['channels'][-1]
            self.struct_encoder = StructEncoder(batch_norm=batch_norm, dropout=dropout, **cdconv_param)
        else:
            struct_dim = 0
        self.struct_dim = struct_dim

        # modal fusion
        """
        proj_dim: try which is best.
            256: ec_mulpro_noseq-20240403-T20-17, 0.9019
            512: ec_mulpro_noseq-20240403-T20-19, 0.9026
            768: ec_mulpro_noseq-20240403-T10-53, 0.9049
            768: ec_mulpro_noseq-20240408-T20-49, 0.9023
            768: ec_mulpro_noseq-20240409-T20-29, 0.9037
            1024: ec_mulpro_noseq-20240403-T20-21, 0.9051
        """
        concat_dim = 0

        # cdconv model embeddings
        if struct_dim > 0:
            self.projector_struct = Projector(in_channels=struct_dim, out_channels=feat_dim)
            concat_dim += feat_dim
        
        # ESM-2 model embeddings
        if seq_emb_dim > 0:
            self.projector_seq = Projector(in_channels=seq_emb_dim, out_channels=feat_dim)
            concat_dim += feat_dim
            
        # domain embeddings
        if self.domain_dim > 0:
            concat_dim += feat_dim
            if not fad_emb: # FAD embeddings or learned embeddings
                assert domain_num > 0, "domain_num should be greater than 0 when not using FAD embeddings"
                self.domain_embeddings = torch.nn.Embedding(num_embeddings=domain_num, embedding_dim=domain_dim)

            if domain_attention: # domain attention or meaning of domains
                self.domain_attention = DomainAttention(n_embd=domain_dim, domain_len=max_domain_num, pos_emb='box')
            else:
                self.domain_attention = DomainMeaning()
            #self.projector_domain = Projector(in_channels=domain_dim, out_channels=feat_dim, dropout=0.2) 
            # may be overfitting, set dropout to 0.2? go_bp_mulpro_noseq-20240422-T11-00, 0.4907
            #self.projector_domain = Projector(in_channels=domain_dim, out_channels=feat_dim)
            self.projector_domain = Projector(in_channels=domain_dim, out_channels=feat_dim)

        if self.contrastive_loss: # protein cropping multimodal contrastive representation
            self.temp = nn.Parameter(torch.ones([]) * 0.07) # MCR:1/100
            self.projector_contrast = Projector(in_channels=feat_dim, out_channels=feat_dim)
        
        self.classifier = MLP(in_channels=concat_dim,
                              mid_channels=max(feat_dim, num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

    def subdomains(self, domain_num, domain_embs, domain_poss, domain_ids):
        domain_num = torch.div(domain_num, 2, rounding_mode='floor')
        if not self.fad_emb:
            domain_embs = self.domain_embeddings(domain_ids)
        domain = self.domain_attention(domain_embs, domain_num, domain_poss)
        domain_feature = self.projector_domain(domain)
        return domain_feature

    def forward(self, data, all_loss=None):
        x, pos, seq, ori, seq_emb, domain_num, domain_embs, domain_poss, domain_ids, batch = data.x, data.pos, data.seq, data.ori, data.seq_emb, data.domain_num, data.domain_embs, data.domain_poss, data.domain_ids, data.batch

        out = torch.zeros((domain_num.shape[0], 0), device=x.device)

        # structure feature
        if self.struct_dim > 0:
            struct_feature = self.projector_struct(self.struct_encoder(x, pos, seq, ori, batch))
            out = torch.cat([out, struct_feature], dim=-1)

        # sequence feature
        if self.seq_emb_dim > 0:
            seq_feature = self.projector_seq(seq_emb)
            out = torch.cat([out, seq_feature], dim=-1)

        # domain feature
        if self.domain_dim > 0:
            if not self.fad_emb:
                domain_embs = self.domain_embeddings(domain_ids)
            domain = self.domain_attention(domain_embs, domain_num, domain_poss)
            domain_feature = self.projector_domain(domain)
            out = torch.cat([out, domain_feature], dim=-1)

        # protein cropping multimodal contrastive representation
        if self.contrastive_loss and (all_loss is not None):
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)

            # crop domain view
            subdomain_feature = self.subdomains(domain_num, domain_embs, domain_poss, domain_ids)
            subdomain_embed = self.projector_contrast(
                    F.normalize(subdomain_feature + 0.1 * torch.randn_like(subdomain_feature, dtype=subdomain_feature.dtype, device=subdomain_feature.device), dim=-1, p=2)
                ) # MCR: noise sigma: 0.004
            subdomain_embed_norm = F.normalize(subdomain_embed, dim=-1, p=2)

            # whole domain view
            domain_embed = self.projector_contrast(
                    F.normalize(domain_feature + 0.1 * torch.randn_like(domain_feature, dtype=domain_feature.dtype, device=domain_feature.device), dim=-1, p=2)
                )
            domain_embed_norm = F.normalize(domain_embed, dim=-1, p=2)

            if self.struct_dim > 0:
                struct_embed = self.projector_contrast(
                    F.normalize(struct_feature + 0.1 * torch.randn_like(struct_feature, dtype=struct_feature.dtype, device=struct_feature.device), dim=-1, p=2)
                )
                struct_embed_norm = F.normalize(struct_embed, dim=-1, p=2)

                if self.protein_cropping:
                    # triplet loss: archor (structure), positive (domain), negative (subdomain)
                    all_loss += torch.mean(-F.cosine_similarity(struct_embed_norm, domain_embed_norm, dim=-1)
                                            + 0.5 * F.relu(-F.cosine_similarity(struct_embed_norm, domain_embed_norm.detach(), dim=-1) + F.cosine_similarity(struct_embed_norm, subdomain_embed_norm.detach(), dim=-1) + 0.1) 
                                            ) 
                    # 加了个lambda，来测试能不能减小gap，
                    # lambda=0.1: ec_mulpro_cl_noseq-20240418-T22-04, 0.9046
                    # lambda=0.5: ec_mulpro_cl_noseq-20240419-T20-53, 0.9070
                    # lambda=1.0: ec_mulpro_cl_noseq-20240411-T20-45, 0.9061
                else:
                    # cosine similarity as logits
                    sim_d2t = domain_embed_norm @ struct_embed_norm.t().detach() / self.temp
                    sim_t2d = struct_embed_norm @ domain_embed_norm.t().detach() / self.temp
                    loss_d2t = -torch.sum(F.log_softmax(sim_d2t, dim=1), dim=1).mean()
                    loss_t2d = -torch.sum(F.log_softmax(sim_t2d, dim=1), dim=1).mean()
                    all_loss += (loss_t2d + loss_d2t) / 2

            if self.seq_emb_dim > 0:
                seq_embed = self.projector_contrast(
                    F.normalize(seq_feature + 0.1 * torch.randn_like(seq_feature, dtype=seq_feature.dtype, device=seq_feature.device), dim=-1, p=2)
                )
                seq_embed_norm = F.normalize(seq_embed, dim=-1, p=2)

                if self.protein_cropping:
                    # triplet loss: archor (sequence), positive (domain), negative (subdomain)
                    all_loss += torch.mean(-F.cosine_similarity(seq_embed_norm, domain_embed_norm, dim=-1)
                                            + 0.5 * F.relu(-F.cosine_similarity(seq_embed_norm, domain_embed_norm.detach(), dim=-1) + F.cosine_similarity(seq_embed_norm, subdomain_embed_norm.detach(), dim=-1) + 0.1) 
                                            )
                else:
                    # cosine similarity as logits
                    sim_d2q = domain_embed_norm @ seq_embed_norm.t().detach() / self.temp
                    sim_q2d = seq_embed_norm @ domain_embed_norm.t().detach() / self.temp
                    loss_d2q = -torch.sum(F.log_softmax(sim_d2q, dim=1), dim=1).mean()
                    loss_q2d = -torch.sum(F.log_softmax(sim_q2d, dim=1), dim=1).mean()
                    all_loss += (loss_d2q + loss_q2d) / 2
                
        out = self.classifier(out)
        return out
    
    def get_loss(self, data, loss_fn):
        device = next(self.parameters()).device
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)

        all_loss = torch.tensor(0, dtype=torch.float32, device=device)
        loss = loss_fn(self.forward(data, all_loss).sigmoid(), y)
        all_loss += loss

        return all_loss
    
    def get_feature(self, data):
        x, pos, seq, ori, seq_emb, domain_num, domain_embs, domain_poss, domain_ids, batch = data.x, data.pos, data.seq, data.ori, data.seq_emb, data.domain_num, data.domain_embs, data.domain_poss, data.domain_ids, data.batch

        # structure feature
        if self.struct_dim > 0:
            struct_feature = self.projector_struct(self.struct_encoder(x, pos, seq, ori, batch))

        # sequence feature
        if self.seq_emb_dim > 0:
            seq_feature = self.projector_seq(seq_emb)

        # domain feature
        if self.domain_dim > 0:
            if not self.fad_emb:
                domain_embs = self.domain_embeddings(domain_ids)
            domain = self.domain_attention(domain_embs, domain_num, domain_poss)
            domain_feature = self.projector_domain(domain)
                
        return seq_feature, struct_feature, domain_feature
    

    
class Model_MVAE(nn.Module):
    def __init__(self,
                 geometric_radii: List,
                 sequential_kernel_size: float,
                 kernel_channels: List,
                 channels: List,
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 seq_emb_dim: int = 0,
                 domain_dim: int = 0,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()
        self.seq_emb_dim = seq_emb_dim
        self.domain_dim = domain_dim

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)
        
        # modal fusion
        self.proj_dim = proj_dim = max(channels[-1], seq_emb_dim, domain_dim)
        self.projector_PoE = PoE_Projector(x_channels=[channels[-1], seq_emb_dim, domain_dim], z_channels=proj_dim)
        
        self.classifier = MLP(in_channels=proj_dim,
                              mid_channels=max(proj_dim, num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

    def forward(self, data, all_loss=None):
        
        x, pos, seq, ori, domain, seq_emb, batch = (self.embedding(data.x), data.pos, data.seq, data.ori, data.domain, data.seq_emb, data.batch)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)

        if all_loss is not None:
            # 以50%的概率不使用domain
            if torch.rand(1) < 0.5:
                out, _ = self.projector_PoE([x, seq_emb, None], all_loss)
            else:
                out, _ = self.projector_PoE([x, seq_emb, domain], all_loss)
        else:
            domain_index = domain.sum(dim=-1, keepdim=False).bool()
            _, out = self.projector_PoE([x, seq_emb, None])
            _, out[domain_index] = self.projector_PoE([x[domain_index], seq_emb[domain_index], domain[domain_index]])
        
        out = self.classifier(out)
        return out




class Model_MUSE(nn.Module):
    def __init__(self,
                 geometric_radii: List,
                 sequential_kernel_size: float,
                 kernel_channels: List,
                 channels: List,
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 seq_emb_dim: int = 0,
                 domain_dim: int = 0,
                 contrastive_loss: bool = False,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()
        self.seq_emb_dim = seq_emb_dim
        self.domain_dim = domain_dim
        self.contrastive_loss = contrastive_loss

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     #kernel_channels = list(kernel_channels),
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     #kernel_channels = list(kernel_channels),
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)
        
        # modal fusion
        proj_dim = max(channels[-1], seq_emb_dim, domain_dim)
        self.projector_muse = MUSE_Projector(x_channels=[channels[-1], seq_emb_dim, domain_dim], z_channels=proj_dim)
        concat_dim = 3 * proj_dim

        if self.contrastive_loss:
            self.temp = nn.Parameter(torch.ones([]) * 0.07)
            self.projector_contrast = Projector(in_channels=proj_dim, out_channels=proj_dim)
        
        self.classifier = MLP(in_channels=concat_dim,
                              mid_channels=max(proj_dim, num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

    def forward(self, data, all_loss=None):
        x, pos, seq, ori, domain, seq_emb, batch = (self.embedding(data.x), data.pos, data.seq, data.ori, data.domain, data.seq_emb, data.batch)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
                
        if all_loss is not None:
            # 以50%的概率不使用domain
            if torch.rand(1) < 0.5:
                zs = self.projector_muse([x, seq_emb, None], all_loss)
            else:
                zs = self.projector_muse([x, seq_emb, domain], all_loss)
            struct_feature, seq_feature, domain_feature = zs
        else:
            domain_index = domain.sum(dim=-1, keepdim=False).bool()
            struct_feature, seq_feature, domain_feature = self.projector_muse([x, seq_emb, None])
            struct_feature[domain_index], seq_feature[domain_index], domain_feature[domain_index] = self.projector_muse([x[domain_index], seq_emb[domain_index], domain[domain_index]])

        out = torch.cat([struct_feature, seq_feature, domain_feature], dim=-1)

        if self.contrastive_loss and all_loss is not None:
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)

            struct_embed = self.projector_contrast(
                F.normalize(struct_feature + 0.1 * torch.randn_like(struct_feature, dtype=struct_feature.dtype, device=struct_feature.device), dim=-1, p=2)
            )
            struct_embed = F.normalize(struct_embed, dim=-1, p=2)
            if self.seq_emb_dim > 0:
                seq_embed = self.projector_contrast(
                    F.normalize(seq_feature + 0.1 * torch.randn_like(seq_feature, dtype=seq_feature.dtype, device=seq_feature.device), dim=-1, p=2)
                )
                seq_embed = F.normalize(seq_embed, dim=-1, p=2)
                # cosine similarity as logits
                sim_t2q = struct_embed @ seq_embed.t().detach() / self.temp
                sim_q2t = seq_embed @ struct_embed.t().detach() / self.temp
                loss_t2q = -torch.sum(F.log_softmax(sim_t2q, dim=1), dim=1).mean()
                loss_q2t = -torch.sum(F.log_softmax(sim_q2t, dim=1), dim=1).mean()
                all_loss += (loss_t2q + loss_q2t) / 2
            if self.domain_dim > 0:
                domain_embed = self.projector_contrast(
                    F.normalize(domain_feature + 0.1 * torch.randn_like(domain_feature, dtype=domain_feature.dtype, device=domain_feature.device), dim=-1, p=2)
                )
                domain_embed = F.normalize(domain_embed, dim=-1, p=2)
                # cosine similarity as logits
                sim_t2d = struct_embed @ domain_embed.t().detach() / self.temp
                sim_d2t = domain_embed @ struct_embed.t().detach() / self.temp
                loss_t2d = -torch.sum(F.log_softmax(sim_t2d, dim=1), dim=1).mean()
                loss_d2t = -torch.sum(F.log_softmax(sim_d2t, dim=1), dim=1).mean()
                all_loss += (loss_t2d + loss_d2t) / 2
                
        out = self.classifier(out)
        return out
    
    def get_loss(self, data, loss_fn):
        device = next(self.parameters()).device
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)

        all_loss = torch.tensor(0, dtype=torch.float32, device=device)
        loss = loss_fn(self.forward(data, all_loss).sigmoid(), y)
        all_loss += loss

        return all_loss




class Model_GMC(nn.Module):
    def __init__(self,
                 geometric_radii: List,
                 sequential_kernel_size: float,
                 kernel_channels: List,
                 channels: List,
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 seq_emb_dim: int = 0,
                 domain_dim: int = 0,
                 contrastive_loss: bool = False,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        super().__init__()

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=embedding_dim)
        self.local_mean_pool = AvgPooling()
        self.seq_emb_dim = seq_emb_dim
        self.domain_dim = domain_dim
        self.use_contrastive = contrastive_loss

        layers = []
        in_channels = embedding_dim
        for i, radius in enumerate(geometric_radii):
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels,
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            layers.append(BasicBlock(r = radius,
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels[i],
                                     out_channels = channels[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))
            in_channels = channels[i]

        self.layers = nn.Sequential(*layers)
        
        # modal fusion
        proj_dim = max(channels[-1], seq_emb_dim, domain_dim)
        self.projector_struct = Projector(in_channels=channels[-1], out_channels=proj_dim)
        concat_dim = proj_dim
        
        # ESM-2 model embeddings
        if seq_emb_dim > 0:
            self.projector_seq = Projector(in_channels=seq_emb_dim, out_channels=proj_dim)
            concat_dim += proj_dim
            
        # domain embeddings
        if domain_dim > 0:
            self.projector_domain = Projector(in_channels=domain_dim, out_channels=proj_dim)
            concat_dim += proj_dim

        if self.use_contrastive:
            self.temp = nn.Parameter(torch.ones([]) * 0.07)
            self.projector_contrast = Projector(in_channels=proj_dim, out_channels=proj_dim)

        self.complete_projector = Projector(in_channels=concat_dim, out_channels=proj_dim)
        self.struct_seq_projector = Projector(in_channels=proj_dim * 2, out_channels=proj_dim)
        self.temp_gmc = nn.Parameter(torch.ones([]) * 0.07)
        self.projector_gmc = Projector(in_channels=proj_dim, out_channels=proj_dim)
        
        self.classifier = MLP(in_channels=proj_dim,
                              mid_channels=max(proj_dim, num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)
        self.classifier2 = MLP(in_channels=proj_dim,
                              mid_channels=max(proj_dim, num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

    def forward(self, data, all_loss=None):
        
        #x, pos, seq, ori, batch = (self.embedding(data.x), data.pos, data.seq, data.ori, data.batch)
        x, pos, seq, ori, domain, seq_emb, batch = (self.embedding(data.x), data.pos, data.seq, data.ori, data.domain, data.seq_emb, data.batch)

        for i, layer in enumerate(self.layers):
            x = layer(x, pos, seq, ori, batch)
            if i == len(self.layers) - 1:
                x = global_mean_pool(x, batch)
            elif i % 2 == 1:
                x, pos, seq, ori, batch = self.local_mean_pool(x, pos, seq, ori, batch)
                
        struct_feature = self.projector_struct(x)
        # sequence feature
        seq_feature = self.projector_seq(seq_emb)
        # domain feature
        domain_feature = self.projector_domain(domain)

        if all_loss is not None:
            if self.use_contrastive:
                with torch.no_grad():
                    self.temp.clamp_(0.001,0.5)

                struct_embed = self.projector_contrast(
                    F.normalize(struct_feature + 0.1 * torch.randn_like(struct_feature, dtype=struct_feature.dtype, device=struct_feature.device), dim=-1, p=2)
                )
                struct_embed = F.normalize(struct_embed, dim=-1, p=2)
                if self.seq_emb_dim > 0:
                    seq_embed = self.projector_contrast(
                        F.normalize(seq_feature + 0.1 * torch.randn_like(seq_feature, dtype=seq_feature.dtype, device=seq_feature.device), dim=-1, p=2)
                    )
                    seq_embed = F.normalize(seq_embed, dim=-1, p=2)
                    # cosine similarity as logits
                    sim_t2q = struct_embed @ seq_embed.t().detach() / self.temp
                    sim_q2t = seq_embed @ struct_embed.t().detach() / self.temp
                    loss_t2q = -torch.sum(F.log_softmax(sim_t2q, dim=1), dim=1).mean()
                    loss_q2t = -torch.sum(F.log_softmax(sim_q2t, dim=1), dim=1).mean()
                    all_loss += (loss_t2q + loss_q2t) / 2
                if self.domain_dim > 0:
                    domain_embed = self.projector_contrast(
                        F.normalize(domain_feature + 0.1 * torch.randn_like(domain_feature, dtype=domain_feature.dtype, device=domain_feature.device), dim=-1, p=2)
                    )
                    domain_embed = F.normalize(domain_embed, dim=-1, p=2)
                    # cosine similarity as logits
                    sim_t2d = struct_embed @ domain_embed.t().detach() / self.temp
                    sim_d2t = domain_embed @ struct_embed.t().detach() / self.temp
                    loss_t2d = -torch.sum(F.log_softmax(sim_t2d, dim=1), dim=1).mean()
                    loss_d2t = -torch.sum(F.log_softmax(sim_d2t, dim=1), dim=1).mean()
                    all_loss += (loss_t2d + loss_d2t) / 2

            complete_fea = self.complete_projector(torch.cat([struct_feature, seq_feature, domain_feature], dim=-1))
            incomplete_fea = self.struct_seq_projector(torch.cat([struct_feature, seq_feature], dim=-1))
            complete_emb = self.projector_gmc(complete_fea)
            incomplete_emb = self.projector_gmc(incomplete_fea)
            all_loss += self.contrastive_loss(complete_emb, incomplete_emb, self.temp_gmc)

            return self.classifier(complete_fea), self.classifier2(incomplete_fea)
        
        else:
            domain_index = domain.sum(dim=-1, keepdim=False).bool()
            out = self.classifier2(self.struct_seq_projector(torch.cat([struct_feature, seq_feature], dim=-1)))
            out[domain_index] = self.classifier(self.complete_projector(torch.cat([struct_feature[domain_index], seq_feature[domain_index], domain_feature[domain_index]], dim=-1)))
            return out
    
    def contrastive_loss(self, emb1, emb2, temp):
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
    
    def get_loss(self, data, loss_fn):
        device = next(self.parameters()).device
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)

        all_loss = torch.tensor(0, dtype=torch.float32, device=device)
        pred1, pred2 = self.forward(data, all_loss)
        all_loss += loss_fn(pred1.sigmoid(), y) + loss_fn(pred2.sigmoid(), y)
        return all_loss



