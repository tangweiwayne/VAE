import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        # 确保目录存在，否则 glob 可能为空
        self.paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(root=config.data_root, transform=transform)
    loader = DataLoader(dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True,
                        num_workers=8, 
                        pin_memory=True)
    return loader, dataset
