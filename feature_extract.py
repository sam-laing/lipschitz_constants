
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. Load CIFAR-10
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='/fast/slaing/data/vision/cifar10/', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# 2. Load pretrained ResNet-18
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.eval()  # inference mode
resnet18.to(device)

# Remove the final classification layer
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])  # outputs (batch_size, 512, 1, 1)

# 3. Extract features
features = []
labels = []

with torch.no_grad():
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        x = feature_extractor(images)  # (batch_size, 512, 1, 1)
        x = x.view(x.size(0), -1)      # flatten to (batch_size, 512)
        features.append(x)
        labels.append(torch.nn.functional.one_hot(targets, num_classes=10))

features = torch.cat(features, dim=0)  # (50000, 512)
labels = torch.cat(labels, dim=0)      # (50000, 10)


if __name__ == "__main__":
    # Save features and labels for later use to path
    path = "/fast/slaing/pretrained_features/cifar10_resnet18_features.pt"
    torch.save({'features': features, 'labels': labels}, path)
    print(f"Feature extraction complete. Features saved to '{path}'.")


