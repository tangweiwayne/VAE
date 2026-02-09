import torch
import os

# ========== Paths ==========
# 建议根据实际环境修改 data_root
data_root = "/Users/wayne/Downloads/train_data/faces_vae_ddpm/img_align_celeba/img_align_celeba"
save_dir = "./vae_test_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# ========== Hyperparams ==========
batch_size = 1024
lr = 2e-4
num_epochs = 300
latent_dim = 128
sample_every = 5
num_sample_images = 8
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
