import torch
from torchvision import utils

def save_checkpoint(model, optim, epoch, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }
    torch.save(state, path)

def save_image_grid(tensor, filename, nrow=8):
    tensor = torch.clamp(tensor, 0, 1)
    utils.save_image(tensor, filename, nrow=nrow, padding=2)
