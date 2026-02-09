import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    # 使用 MSE 重建损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld
