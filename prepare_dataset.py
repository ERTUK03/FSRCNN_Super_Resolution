from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None,):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, file) for file in os.listdir(root_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        lr_image = transforms.functional.resize(image, (162, 162))
        return lr_image, image

def get_dataloaders(dir: str,
                    batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop((647, 647)),
        transforms.RandomHorizontalFlip(p=0.5)])

    dataset = CustomImageDataset(root_dir=dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
