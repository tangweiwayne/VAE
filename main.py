import torch
import os
from datetime import datetime
import config
from model import ConvVAE
from dataset import get_dataloader
from loss import vae_loss
from utils import save_checkpoint, save_image_grid

def train():
    # 获取数据
    loader, dataset = get_dataloader()
    
    # 初始化模型
    model = ConvVAE(latent_dim=config.latent_dim, image_size=config.image_size).to(config.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    global_step = 0
    print("Starting training on device:", config.device)
    print("Dataset size:", len(dataset))

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0

        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(config.device, non_blocking=True)
            optim.zero_grad()
            recon_imgs, mu, logvar = model(imgs)
            loss, recon_l, kld = vae_loss(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kld += kld.item()
            global_step += 1

            if batch_idx % 100 == 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Epoch {epoch}/{config.num_epochs} Batch {batch_idx}/{len(loader)} "
                    f"Loss {loss.item():.4f} "
                    f"(recon {recon_l.item():.4f}, kld {kld.item():.4f})")

        n_samples = len(loader.dataset)
        if n_samples > 0:
            avg_loss = epoch_loss / n_samples
            avg_recon = epoch_recon / n_samples
            avg_kld = epoch_kld / n_samples
        else:
            avg_loss = avg_recon = avg_kld = 0.0

        print(f"=== Epoch {epoch} finished. Avg loss: {avg_loss:.4f} "
            f"(recon {avg_recon:.4f}, kld {avg_kld:.4f}) ===")

        # 保存样本
        if epoch % config.sample_every == 0 or epoch == 1:
            # 保存检查点
            ckpt_path = os.path.join(config.save_dir, f"vae_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path)
            model.eval()
            with torch.no_grad():
                # 重建样本
                # 注意：如果 dataset 为空，这里会报错，实际使用需保证有数据
                try:
                    imgs = next(iter(loader))
                    imgs = imgs.to(config.device)[:config.num_sample_images]
                    recon_imgs, _, _ = model(imgs)
                    combined = torch.cat([imgs, recon_imgs], dim=0)
                    save_image_grid(combined, os.path.join(config.save_dir, f"recon_epoch{epoch}.png"), nrow=8)

                    # 生成样本
                    z = torch.randn(config.num_sample_images, config.latent_dim).to(config.device)
                    samples = model.decode(z)
                    save_image_grid(samples, os.path.join(config.save_dir, f"sample_epoch{epoch}.png"), nrow=8)
                except StopIteration:
                    print("Warning: DataLoader is empty, cannot save samples.")
                    
            model.train()

    print("Training complete.")

if __name__ == "__main__":
    train()
