import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(
            self, num_classes=5,  hidden_channels=64,
            dropout_rate=0.25
            ):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, int(hidden_channels//2), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(int(hidden_channels//2), hidden_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(hidden_channels * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    model = SimpleCNN(num_classes=5)
    print(model)
    sample_input = torch.randn(1, 3, 32, 32)
    output = model(sample_input)
    print("Output shape:", output.shape)  # should be [1, 5]