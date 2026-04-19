import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net, PrunableLinear
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss

lambda_val = 0.001

for epoch in range(3):
    for images, labels in train_loader:
        outputs = model(images)
        loss1 = criterion(outputs, labels)
        loss2 = sparsity_loss(model)
        loss = loss1 + lambda_val * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training done")

def calculate_sparsity(model):
    total = 0
    zero = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < 1e-2).sum().item()
    return 100 * zero / total

print("Sparsity:", calculate_sparsity(model))

for module in model.modules():
    if isinstance(module, PrunableLinear):
        gates = torch.sigmoid(module.gate_scores).detach().numpy()
        plt.hist(gates.flatten(), bins=50)
        plt.savefig("results.png")
        break