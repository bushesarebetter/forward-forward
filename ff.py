import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_CLASSES = 10


def load_data(seed=0):
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    return torch.tensor(Xtr), torch.tensor(Xte), torch.tensor(ytr), torch.tensor(yte)


def embed(x, labels):
    # overwrite the first 10 pixels with a one-hot label (they're the top-left corner, almost always blank)
    x = x.clone()
    x[:, :N_CLASSES] = 0.0
    x[torch.arange(len(x)), labels] = 1.0
    return x


def goodness(h):
    return h.pow(2).sum(dim=1)


def ff_loss(g_pos, g_neg, theta):
    # logistic loss: p(positive) = sigmoid(g - theta); push pos above theta, neg below
    return torch.cat([F.softplus(theta - g_pos), F.softplus(g_neg - theta)]).mean()


class FFLayer(nn.Linear):
    def forward(self, x):
        # normalise length so the next layer only sees the direction, not the previous goodness
        x = x / (x.norm(dim=1, keepdim=True) + 1e-4)
        return F.relu(super().forward(x))


class FFNet(nn.Module):
    def __init__(self, sizes, theta, lr):
        super().__init__()
        self.theta = theta
        self.layers = nn.ModuleList(FFLayer(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.opts = [torch.optim.Adam(layer.parameters(), lr=lr) for layer in self.layers]

    def layer_losses(self, x_pos, x_neg):
        losses = []
        for layer in self.layers:
            h_pos, h_neg = layer(x_pos), layer(x_neg)
            losses.append(ff_loss(goodness(h_pos), goodness(h_neg), self.theta))
            x_pos, x_neg = h_pos.detach(), h_neg.detach()  # no gradient between layers
        return losses

    def train_step(self, x, y):
        wrong = (y + torch.randint(1, N_CLASSES, y.shape)) % N_CLASSES
        losses = self.layer_losses(embed(x, y), embed(x, wrong))
        for opt in self.opts:
            opt.zero_grad()
        sum(losses).backward()  # losses are independent, so this is per-layer training
        for opt in self.opts:
            opt.step()

    @torch.no_grad()
    def predict(self, x):
        scores = torch.zeros(len(x), N_CLASSES)
        for label in range(N_CLASSES):
            h = embed(x, torch.full((len(x),), label))
            for layer in self.layers:
                h = layer(h)
                scores[:, label] += goodness(h)
        return scores.argmax(dim=1)


def mlp(sizes):
    mods = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*mods[:-1])


def accuracy(pred, y):
    return (pred == y).float().mean().item()


def run(hidden=500, n_layers=2, epochs=40, batch=64, theta=2.0, lr=0.003, seed=0):
    torch.manual_seed(seed)
    Xtr, Xte, ytr, yte = load_data(seed)
    ff = FFNet([64] + [hidden] * n_layers, theta=theta, lr=lr)
    bp = mlp([64] + [hidden] * n_layers + [N_CLASSES])
    bp_opt = torch.optim.Adam(bp.parameters(), lr=1e-3)

    hist = {"ff": [], "bp": []}
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            ff.train_step(Xtr[idx], ytr[idx])
            bp_opt.zero_grad()
            F.cross_entropy(bp(Xtr[idx]), ytr[idx]).backward()
            bp_opt.step()
        with torch.no_grad():
            hist["ff"].append(accuracy(ff.predict(Xte), yte))
            hist["bp"].append(accuracy(bp(Xte).argmax(1), yte))
    with torch.no_grad():
        hist["ff_train"] = accuracy(ff.predict(Xtr), ytr)
        hist["bp_train"] = accuracy(bp(Xtr).argmax(1), ytr)
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=500)
    args = p.parse_args()
    epochs, hidden = (5, 100) if args.quick else (args.epochs, args.hidden)

    hist = run(hidden=hidden, epochs=epochs)
    print(f"forward-forward  train {hist['ff_train']:.3f}  test {hist['ff'][-1]:.3f}")
    print(f"backprop MLP     train {hist['bp_train']:.3f}  test {hist['bp'][-1]:.3f}")

    ep = np.arange(1, epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, hist["ff"], label=f"forward-forward ({hist['ff'][-1]:.3f})")
    plt.plot(ep, hist["bp"], label=f"backprop ({hist['bp'][-1]:.3f})")
    plt.axhline(0.1, color="gray", ls=":", label="chance")
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.title(f"load_digits, 64-{hidden}-{hidden}")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("demo.png", dpi=120)


if __name__ == "__main__":
    main()
