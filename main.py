import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Custom Layer ─────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))  # sigmoid(0)=0.5

    def forward(self, x):
        gates        = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

# ── Model ────────────────────────────────────────────────────────────────────
class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 256)
        self.fc2 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten 32×32×3 → 3072
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ── Sparsity Loss ─────────────────────────────────────────────────────────────
def sparsity_loss(model):
    total = 0.0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            total = total + torch.sigmoid(m.gate_scores).sum()
    return total

# ── Sparsity % (gates < 1e-2) ─────────────────────────────────────────────────
def compute_sparsity(model):
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu()
            all_gates.append(gates.flatten())
    all_gates = torch.cat(all_gates)
    pct = (all_gates < 1e-2).float().mean().item() * 100
    return pct, all_gates.numpy()

# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ── Train ─────────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, lam):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = F.cross_entropy(logits, labels) + lam * sparsity_loss(model)
        loss.backward()
        optimizer.step()

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds   = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_loader, test_loader = get_loaders()
    lambdas   = [0, 1e-5, 1e-4]
    results   = []
    all_gates = {}

    for lam in lambdas:
        print(f"\n── Training with λ={lam} ──")
        model     = PrunableMLP().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(3):
            train(model, train_loader, optimizer, lam)
            acc = evaluate(model, test_loader)
            print(f"  Epoch {epoch+1}/3 | Test Acc: {acc:.2f}%")

        acc              = evaluate(model, test_loader)
        sparsity, gates  = compute_sparsity(model)
        all_gates[lam]   = gates
        results.append((lam, acc, sparsity))
        print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Final Results ──")
    for lam, acc, sparsity in results:
        print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

    # ── Histogram ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Gate Value Distributions by λ", fontsize=14, fontweight="bold")

    for ax, (lam, acc, sparsity) in zip(axes, results):
        gates = all_gates[lam]
        ax.hist(gates, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(1e-2, color="red", linestyle="--", linewidth=1.2, label="threshold (0.01)")
        ax.set_title(f"λ={lam}\nAcc={acc:.1f}%  Sparsity={sparsity:.1f}%")
        ax.set_xlabel("Gate Value (sigmoid)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("gate_histograms.png", dpi=150)
    print("\nHistogram saved → gate_histograms.png")
    plt.show()

if __name__ == "__main__":
    main()