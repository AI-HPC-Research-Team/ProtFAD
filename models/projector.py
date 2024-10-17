import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 res_expansion: float = 2.0,
                 dropout: float = 0.0,
                 bias: bool = True) -> nn.Module:
        super(Projector, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.net1 = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * res_expansion), bias=bias),
            nn.BatchNorm1d(int(in_channels * res_expansion)),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            #nn.Dropout(dropout),
            nn.Linear(int(in_channels * res_expansion), out_channels, bias=bias),
            nn.BatchNorm1d(out_channels)
        )
        self.init_weights('eye')
        
    def get_device(self):
        return next(self.parameters()).device
    
    def init_weights(self, mode):
        # initialize transformer
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)

    def forward(self, x):
        return self.net2(self.net1(x))

class VAE_Projector(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int = None, 
                 res_expansion: float = 2.0,
                 bias: bool = True) -> nn.Module:
        super(VAE_Projector, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        self.mu_encoder = Projector(in_channels=in_channels, out_channels=out_channels, res_expansion=res_expansion, bias=bias)
        self.logvar_encoder = Projector(in_channels=in_channels, out_channels=out_channels, res_expansion=res_expansion, bias=bias)

        self.decoder = Projector(in_channels=out_channels, out_channels=in_channels, res_expansion=res_expansion, bias=bias)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, all_loss=None):
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z)

        if all_loss is not None:
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            recon_loss = F.mse_loss(recon_x, x)
            all_loss += kl_loss + recon_loss

        return z # 这里用z还是用mu比较好
    

from .quantize import VectorQuantizer
class VQ_Projector(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int = None, 
                 res_expansion: float = 2.0,
                 bias: bool = True) -> nn.Module:
        super(VQ_Projector, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        discrete_channels = 128

        self.encoder = Projector(in_channels=in_channels, out_channels=out_channels, res_expansion=res_expansion, bias=bias)
        self.codebook = VectorQuantizer(n_e=1024, e_dim=discrete_channels)
        self.decoder = Projector(in_channels=out_channels, out_channels=in_channels, res_expansion=1/res_expansion, bias=bias)

    def forward(self, x, all_loss=None):
        z = self.encoder(x)        
        z_q, _ = self.codebook(z.view(-1,128), all_loss)
        z_q = z_q.view(z.shape)
        recon_x = self.decoder(z_q)

        if all_loss is not None:
            all_loss += F.mse_loss(recon_x, x)

        return z_q
    

class Domain_Generator(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int = None, 
                 res_expansion: float = 2,
                 bias: bool = True) -> nn.Module:
        super(Domain_Generator, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        latent_channels = int(in_channels / res_expansion)
        self.mu_encoder = Projector(in_channels=in_channels, out_channels=latent_channels, res_expansion=1/res_expansion, bias=bias)
        self.logvar_encoder = Projector(in_channels=in_channels, out_channels=latent_channels, res_expansion=1/res_expansion, bias=bias)

        self.decoder = Projector(in_channels=latent_channels, out_channels=out_channels, res_expansion=res_expansion, bias=bias)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, y=None, all_loss=None, index=None):
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z)

        if all_loss is not None:
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            recon_loss = F.mse_loss(recon_x[index], y[index])
            all_loss += kl_loss + recon_loss

        return recon_x






import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class MVAE_Projector(nn.Module):
    def __init__(self, 
                 x_channels: int, 
                 y_channels: int, 
                 z_channels: int = None, 
                 res_expansion: float = 2.0,
                 bias: bool = True) -> nn.Module:
        super(MVAE_Projector, self).__init__()

        if z_channels is None:
            z_channels = y_channels
        self.z_channels = z_channels

        self.x_mu_encoder = Projector(in_channels=x_channels, out_channels=z_channels, res_expansion=res_expansion, bias=bias)
        self.x_logvar_encoder = Projector(in_channels=x_channels, out_channels=z_channels, res_expansion=res_expansion, bias=bias)
        self.y_mu_encoder = Projector(in_channels=y_channels, out_channels=z_channels, res_expansion=res_expansion, bias=bias)
        self.y_logvar_encoder = Projector(in_channels=y_channels, out_channels=z_channels, res_expansion=res_expansion, bias=bias)
        self.x_decoder = Projector(in_channels=z_channels, out_channels=x_channels, res_expansion=res_expansion, bias=bias)
        self.y_decoder = Projector(in_channels=z_channels, out_channels=y_channels, res_expansion=res_expansion, bias=bias)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x=None, y=None, all_loss=None):
        mu, logvar = self.infer(x, y)
        # reparametrization trick to sample
        z = self.reparameterise(mu, logvar)
        # reconstruct inputs based on that gaussian
        recon_x = self.x_decoder(z)
        recon_y = self.y_decoder(z)

        if all_loss is not None:
            # compute ELBO for each data combo
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            all_loss += kl_loss
            if x is not None:
                all_loss += F.mse_loss(recon_x, x)
            if y is not None:
                all_loss += F.mse_loss(recon_y, y)

        return mu

    def infer(self, x=None, y=None): 
        batch_size = x.size(0) if x is not None else y.size(0)
        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.z_channels), 
                                  use_cuda=use_cuda)
        if x is not None:
            x_mu = self.x_mu_encoder(x)
            x_logvar = self.x_logvar_encoder(x)
            mu     = torch.cat((mu, x_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, x_logvar.unsqueeze(0)), dim=0)

        if y is not None:
            y_mu = self.y_mu_encoder(y)
            y_logvar = self.y_logvar_encoder(y)
            mu     = torch.cat((mu, y_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, y_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.product_of_experts(mu, logvar)
        return mu, logvar
    
    def prior_expert(self, size, use_cuda=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).

        @param size: integer
                    dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                        cast CUDA on variables
        """
        mu     = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar
    
    def product_of_experts(self, mu, logvar, eps=1e-8):
        """Return parameters for product of independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.

        @param mu: M x D for M experts
        @param logvar: M x D for M experts
        """
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class PoE_Projector(nn.Module):
    def __init__(self,
                 x_channels: list,
                 z_channels: int,
                 res_expansion: float = 2.0,
                 bias: bool = True) -> nn.Module:
        super().__init__()

        self.x_channels = x_channels
        self.z_channels = z_channels

        for i, ch in enumerate(x_channels):
            setattr(self, f'x{i}_mu_encoder', Projector(in_channels=ch, out_channels=z_channels, res_expansion=res_expansion, bias=bias))
            setattr(self, f'x{i}_logvar_encoder', Projector(in_channels=ch, out_channels=z_channels, res_expansion=res_expansion, bias=bias))
            setattr(self, f'x{i}_decoder', Projector(in_channels=z_channels, out_channels=ch, res_expansion=res_expansion, bias=bias))

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu, device=mu.device)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, xs, all_loss=None):
        mu, logvar = self.infer(xs)
        # reparametrization trick to sample
        z = self.reparameterise(mu, logvar)

        if all_loss is not None:
            # compute ELBO for each data combo
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            all_loss += kl_loss
            for i, x in enumerate(xs):
                if x is not None:
                    # reconstruct inputs based on that gaussian
                    recon_x = getattr(self, f'x{i}_decoder')(z)
                    all_loss += F.mse_loss(recon_x, x)

        return z, mu
    
    def reconstruct(self, xs, all_loss=None):
        mu, logvar = self.infer(xs)
        # reparametrization trick to sample
        z = self.reparameterise(mu, logvar)

        recon_xs = []
        if all_loss is not None:
            # compute ELBO for each data combo
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            all_loss += kl_loss
        
        for i, x in enumerate(xs):
            # reconstruct inputs based on that gaussian
            recon_x = getattr(self, f'x{i}_decoder')(z)
            recon_xs.append(recon_x)
            if all_loss is not None and x is not None:
                all_loss += F.mse_loss(recon_x, x)

        return recon_xs

    def infer(self, xs):
        assert len(xs) == len(self.x_channels)
        for i, x in enumerate(xs):
            if x is not None:
                batch_size = x.size(0)
                # initialize the universal prior expert
                mu, logvar = self.prior_expert((1, batch_size, self.z_channels), device=x.device)
                break
        for i, x in enumerate(xs):
            if xs[i] is not None:
                temp_mu = getattr(self, f'x{i}_mu_encoder')(x)
                temp_logvar = getattr(self, f'x{i}_logvar_encoder')(x)
                mu     = torch.cat((mu, temp_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, temp_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.product_of_experts(mu, logvar)
        return mu, logvar
    
    def prior_expert(self, size, device='cpu'):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).

        @param size: integer
                    dimensionality of Gaussian
        @param device: cast CUDA on variables
        """
        mu     = Variable(torch.zeros(size)).to(device)
        logvar = Variable(torch.log(torch.ones(size))).to(device)
        return mu, logvar
    
    def product_of_experts(self, mu, logvar, eps=1e-8):
        """Return parameters for product of independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.

        @param mu: M x D for M experts
        @param logvar: M x D for M experts
        """
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar
    


class MUSE_Projector(nn.Module):
    def __init__(self,
                 x_channels: list,
                 z_channels: int,
                 res_expansion: float = 2.0,
                 bias: bool = True) -> nn.Module:
        super().__init__()

        self.x_channels = x_channels
        self.z_channels = z_channels
        self.poe_module = PoE_Projector(x_channels=[z_channels for _ in x_channels], z_channels=z_channels, res_expansion=res_expansion, bias=bias)

        for i, ch in enumerate(x_channels):
            setattr(self, f'x{i}_encoder', VAE_Projector(in_channels=ch, out_channels=z_channels, res_expansion=res_expansion, bias=bias))

    def forward(self, xs, all_loss=None):
        zs = []
        for i, x in enumerate(xs):
            if x is not None:
                z = getattr(self, f'x{i}_encoder')(x, all_loss)
                zs.append(z)
            else:
                zs.append(None)
        recon_zs = self.poe_module.reconstruct(zs, all_loss)
        return recon_zs



