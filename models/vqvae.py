import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import VectorQuantizer

# swish激活函数
class nonlinearity(nn.Module):
    def __init__(self, dropout=False):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = x * torch.sigmoid(x)
        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        return out

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.2):
        super().__init__()

        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        self.net1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nonlinearity(),
            nn.Linear(in_channels, out_channels)
        )
        self.net2 = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nonlinearity(dropout=dropout),
            nn.Linear(out_channels, out_channels)
        )    

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        h = x
        h = self.net2(self.net1(h))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channels, n_channels, z_channels, layer_res_blocks_num, dropout=False, **ignore_kwargs):
        super().__init__()
        self.C = in_channels
        self.nz = z_channels
        self.layer_res_blocks_num = layer_res_blocks_num # 每层模块中含的串行残差块数量
        
        # 输入线性层
        self.linear_in = nn.Linear(in_channels, n_channels[0])
        
        self.net = nn.Sequential()
        for i in range(len(n_channels)-1): # 逐层添加残差块
            block_in, block_out = n_channels[i], n_channels[i+1]
            for j in range(self.layer_res_blocks_num):
                self.net.add_module('res_block_l{}b{}'.format(i,j), ResidualBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out # 确保后面的残差块输入深度和输出深度相等
        
        self.norm_out = nn.Sequential(
            nn.BatchNorm1d(n_channels[-1]),
            nonlinearity(),
            nn.Linear(n_channels[-1], z_channels),
        )

    def forward(self, x):
        return self.norm_out(self.net(self.linear_in(x)))

# 解码器
class Decoder(nn.Module):
    def __init__(self, z_channels, n_channels, out_channels, layer_res_blocks_num, dropout=False, **ignore_kwargs):
        '''resolution：生成图像分辨率'''
        super().__init__()
        self.nz = z_channels
        self.C = out_channels
        self.layer_res_blocks_num = layer_res_blocks_num # 每层模块中含的串行残差块数量

        # 输入线性层
        self.linear_in = nn.Linear(z_channels, n_channels[-1])

        self.net = nn.Sequential()
        for i in reversed(range(len(n_channels)-1)): # 逐层添加残差块
            block_in, block_out = n_channels[i+1], n_channels[i]
            for j in range(self.layer_res_blocks_num+1):
                self.net.add_module('res_block_l{}b{}'.format(i,j), ResidualBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out # 确保后面的残差块输入深度和输出深度相等
        
        # 后处理
        self.norm_out = nn.Sequential(
            nn.BatchNorm1d(n_channels[0]),
            nonlinearity(),
            nn.Linear(n_channels[0], out_channels)
        )

    def forward(self, x, give_pre_end=False):
        out = self.net(self.linear_in(x))
        if not give_pre_end:
            out = self.norm_out(out)
        return out

class VQVAE(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 n_channels: list,
                 z_channels: int,
                 code_size: int,
                 out_channels: int = None,
                 dropout: bool = False) -> nn.Module:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.encoder = Encoder(in_channels=in_channels, n_channels=n_channels, z_channels=z_channels, layer_res_blocks_num=2, dropout=dropout)
        self.codebook = VectorQuantizer(n_e=code_size, e_dim=z_channels)
        self.decoder = Decoder(z_channels=z_channels, n_channels=n_channels, out_channels=out_channels, layer_res_blocks_num=2, dropout=dropout)

    def forward(self, data, all_loss=None):
        x, pos, seq, ori, domain, seq_emb, batch = (data.x, data.pos, data.seq, data.ori, data.domain, data.seq_emb, data.batch)

        z = self.encoder(domain)    
        z_q, _ = self.codebook(z, all_loss)
        recon_domain = self.decoder(z_q)

        if all_loss is not None:
            all_loss += F.mse_loss(recon_domain, domain)

        return recon_domain
    
    def get_latent(self, domain):
        z = self.encoder(domain)
        z_q, _ = self.codebook(z)
        return z, z_q
    