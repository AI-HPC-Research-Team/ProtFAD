import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
import random
from models.loss import ContrastiveLoss, contrastive_loss
from models.projector import Projector

class DomainEmbeddingModel(nn.Module):
    """
    Function-aware domain embedding model
    """
    def __init__(self,
                 num_domains: int,
                 num_goterms: int,
                 emb_dim: int = 256,
                 latent_dim: int = 128,
                 dropout: float = 0.05,
                 loss: str = 'all') -> nn.Module:
        """
        Initializes the model
        Args:
            emb_dim (int): dimension of Embedding. Defaults to 256.
            latent_dim (int, optional): number of neurons in dense layer. Defaults to 128.
            dropout (float, optional): dropout rate. Defaults to 0.05.
            loss(str, optional): all, mse or con. Defaults to all.
        """
        super().__init__()

        self.num_domains = num_domains
        self.num_goterms = num_goterms
        self.emb_dim = emb_dim
        assert loss in ['all', 'mse', 'con'], "loss must be one of ['all', 'mse', 'con']"
        self.loss = loss

        self.embedding_domain = torch.nn.Embedding(num_embeddings=num_domains, embedding_dim=self.emb_dim, max_norm=1)
        self.embedding_go = torch.nn.Embedding(num_embeddings=num_goterms, embedding_dim=self.emb_dim, max_norm=1)

        self.predict_head = nn.Sequential(nn.Linear(emb_dim, latent_dim),
                                          nn.ReLU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(latent_dim, 1),
                                          nn.Sigmoid()
                                          )
        
        #self.text_proj = Projector(768, emb_dim)
        self.project = Projector(emb_dim, emb_dim)
        self.temp = nn.Parameter(torch.ones([]) * 0.07) # MCR:1/100

    def forward(self, data, all_loss=None):
        domain_id, go_id = (data.domain_id, data.go_id)

        domain_embedding = self.embedding_domain(domain_id)
        go_embedding = self.embedding_go(go_id)

        if all_loss is not None and self.loss != 'mse':
            text_embedding = data.text_embedding
            all_loss += contrastive_loss(
                self.project(F.normalize(domain_embedding + 0.001 * torch.randn_like(domain_embedding, dtype=domain_embedding.dtype, device=domain_embedding.device), dim=-1, p=2)),
                self.project(F.normalize(text_embedding, dim=-1, p=2)),
                self.temp)
            """all_loss += contrastive_loss(
                self.project(F.normalize(domain_embedding + 0.001 * torch.randn_like(domain_embedding, dtype=domain_embedding.dtype, device=domain_embedding.device), dim=-1, p=2)),
                self.project(F.normalize(self.text_proj(text_embedding) + 0.001 * torch.randn_like(text_embedding, dtype=text_embedding.dtype, device=text_embedding.device), dim=-1, p=2)),
                self.temp)"""
            """domain_emb = F.normalize(
                self.project(
                    F.normalize(domain_embedding + 0.001 * torch.randn_like(domain_embedding, dtype=domain_embedding.dtype, device=domain_embedding.device), dim=-1, p=2))
                    , dim=-1, p=2)
            text_emb = F.normalize(
                self.project(
                    F.normalize(text_embedding + 0.001 * torch.randn_like(text_embedding, dtype=text_embedding.dtype, device=text_embedding.device), dim=-1, p=2))
                    , dim=-1, p=2)

            similarity = F.cosine_similarity(domain_emb, text_emb, dim=-1)
            all_loss += -torch.mean((data.p.bool() * 2 - 1) * similarity)"""
        
        feat = torch.mul(domain_embedding, go_embedding)
        out = self.predict_head(feat).squeeze(1)

        return out

    def get_domain_embedding(self, domain_id):
        """
        returns the embedding of a domain

        Args:
            domain_id (int): id of the domain

        Returns:
            tensor: domain embedding
        """
        if 0 <= domain_id < self.num_domains:
            return self.embedding_domain(torch.tensor(domain_id)).detach().cpu().numpy()
        else:                         # if domain not found, return 0 default
            return self.embedding_domain(torch.tensor(0)).detach().cpu().numpy() * 0
            
    def get_loss(self, data, loss_fn):
        device = next(self.parameters()).device
        all_loss = torch.tensor(0, dtype=torch.float32, device=device)
        pred = self.forward(data, all_loss)
        if self.loss != 'con':
            all_loss += loss_fn(pred, data.p)
        return all_loss
