import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random

class SVHNOneHotDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.svhn = datasets.SVHN(root=root, download=True, transform=transform)
        self.num_classes = 10  

    def __len__(self):
        return len(self.svhn)

    def __getitem__(self, idx):
        image, label = self.svhn[idx]
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return image, one_hot_label

def get_svhn_loader(subset_size=10_000, random_seed=66):
    """Builds and returns Dataloader for a subset of the SVHN dataset."""
    random.seed(random_seed) 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
        transforms.Lambda(lambda x: x.view(-1))

    ])

    svhn_dataset = SVHNOneHotDataset(root="/fast/slaing/data/vision/svhn/", transform=transform)
    
    #create a subset of the dataset
    indices = list(range(len(svhn_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    svhn_subset = Subset(svhn_dataset, subset_indices)

    svhn_loader = DataLoader(dataset=svhn_subset,
                             batch_size=128,
                             shuffle=True)
    
    return svhn_loader

if __name__ == "__main__":
    svhn_loader = get_svhn_loader()
    print(svhn_loader)
    for images, labels in svhn_loader:
        # check out size of image (without batch dimension)
        B, W, H, C = images.shape
        print("dimension of image",  W*H*C)  # should be 32*32*3=3072

        print(images.shape)
        print(labels.shape)
        break