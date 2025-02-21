import os.path as osp
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
import random
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import torch_geometric.transforms as T
from torch_geometric.nn import fps, global_max_pool, global_mean_pool, radius, global_add_pool

from .cdconv import StructEncoder, BasicBlock
from .projector import *
from .modules import Linear, MLP, DomainAttention, DomainMeaning

class DomainSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, domain_embs, domain_num, domain_poss=None):
        """
        input: domain_embs: [bs, l, 256]
                domain_num: [bs]
                domain_poss: [bs, l], optional
        output: [bs, 256]
        """
        bs, l, dim = domain_embs.shape
        domain_mask = self._create_padding_mask(domain_num, l) # where there is domain, mask it
        domain_embs = domain_embs * domain_mask.unsqueeze(2)

        return domain_embs.sum(dim=1)
    
    def _create_padding_mask(self, length, max_length=16):
        batch_size = length.size(0)                     # 获得batch_size
        seq_range = torch.arange(0, max_length, device=length.device).long()          # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length) # torch.Size([bs, l])
        seq_length_expand = length.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand

class inter_model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(inter_model, self).__init__()
        
        self.domain_embeddings = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.domain_aggregation = DomainSum()

        self.linearLayer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU()
        )
    
    def forward(self, domain_ids, domain_num):
        domain_embs = self.domain_embeddings(domain_ids)
        domain = self.domain_aggregation(domain_embs, domain_num)
        inter_feature = F.relu(domain)
        inter_feature = self.linearLayer(inter_feature)
        return inter_feature

class transformer_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, head=1):
        super(transformer_block, self).__init__()
        self.head = head

        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])

        self.concat_trans = nn.Linear((hidden_dim)*head, hidden_dim, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(in_dim)
    
    def forward(self, residue_h, inter_h, batch):
        multi_output = []
        for i in range(self.head):
            q = self.trans_q_list[i](residue_h)
            k = self.trans_k_list[i](inter_h)
            v = self.trans_v_list[i](residue_h)
            att = torch.sum(torch.mul(q, k)/torch.sqrt(torch.tensor(1280.0)), dim=1, keepdim=True)

            unique_batches = torch.unique(batch)  # 获取所有不同的蛋白质（图）

            tp = torch.zeros_like(v)
            for protein_id in unique_batches:
                node_indices = torch.nonzero(batch == protein_id).reshape(-1)
                att_protein = att[node_indices]
                alpha = F.softmax(att_protein, dim=0).reshape(-1, 1)  # 注意力系数
                tp[node_indices] = v[node_indices] * alpha

            multi_output.append(tp)

        multi_output = torch.cat(multi_output, dim=1)
        multi_output = self.concat_trans(multi_output)

        multi_output = self.layernorm(multi_output + residue_h)

        multi_output = self.layernorm(self.ff(multi_output)+multi_output)

        return multi_output

class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 head: int,
                 r: float,
                 sequential_kernel_size: float,
                 kernel_channels: list,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,) -> nn.Module:

        super(GCN, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv1 = BasicBlock(in_channels = in_dim, out_channels = hidden_dim,
                                r = r, l = sequential_kernel_size, kernel_channels = kernel_channels,
                                base_width = base_width, batch_norm = batch_norm, dropout = dropout, bias = bias)
        self.conv2 = BasicBlock(in_channels = hidden_dim, out_channels = hidden_dim,
                                r = r, l = sequential_kernel_size, kernel_channels = kernel_channels,
                                base_width = base_width, batch_norm = batch_norm, dropout = dropout, bias = bias)

        self.transformer_block = transformer_block(hidden_dim, hidden_dim, head)

    def forward(self, h, pos, seq, ori, batch, inter_f):
        # h: sequence embeddings; g: graph net; inter_f: domain embeddings
        init_avg_h = global_mean_pool(h, batch)

        pre = h
        h = self.bn1(h)
        h = pre + self.dropout(F.relu(self.conv1(h, pos, seq, ori, batch))) # , edge_weight=ew

        pre = h
        h = self.bn2(h)
        h = pre + self.dropout(F.relu(self.conv2(h, pos, seq, ori, batch)))

        residue_h = h
        inter_h = inter_f[batch]
        hg = self.transformer_block(residue_h, inter_h, batch)
        readout = global_add_pool(hg, batch)
        return readout, init_avg_h

class Model(nn.Module):
    def __init__(self,
                 inter_size, inter_hid, graph_size, graph_hid, head,
                 r: float,
                 sequential_kernel_size: float,
                 kernel_channels: list,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.2,
                 num_classes: int = 384) -> nn.Module:
        super().__init__()

        self.inter_embedding = inter_model(inter_size, inter_hid)
        self.GNN = GCN(graph_size, graph_hid, head, r=r, sequential_kernel_size=sequential_kernel_size, kernel_channels=kernel_channels, base_width=base_width, batch_norm=batch_norm, dropout=dropout)
        self.classify = nn.Sequential(
            nn.BatchNorm1d(graph_size + graph_hid),
            nn.Linear(graph_size+graph_hid, (graph_size+graph_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((graph_size+graph_hid)*2, (graph_size+graph_hid)*2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear((graph_size+graph_hid)*2, num_classes)
        )
    
    def forward(self, data, all_loss=None):
        x, pos, seq, ori, seq_emb, domain_num, domain_embs, domain_poss, domain_ids, batch = data.seq_embs, data.pos, data.seq, data.ori, data.seq_emb, data.domain_num, data.domain_embs, data.domain_poss, data.domain_ids, data.batch

        inter_featrue = self.inter_embedding(domain_ids, domain_num)
        graph_feature, init_feature = self.GNN(x, pos, seq, ori, batch, inter_featrue)

        return self.classify(torch.cat((init_feature, graph_feature), 1))

    def get_loss(self, data, loss_fn):
        device = next(self.parameters()).device
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)

        all_loss = torch.tensor(0, dtype=torch.float32, device=device)
        loss = loss_fn(self.forward(data, all_loss).sigmoid(), y)
        all_loss += loss

        return all_loss
