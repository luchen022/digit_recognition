import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import gzip
import struct
import os

# 自动选择 GPU 或 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 1. 数据加载 ---
def load_mnist_local(path='data'):
    def load_file(filepath, is_image):
        with gzip.open(filepath, 'rb') as f:
            if is_image:
                _, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
                return data.astype(np.float32) / 255.0
            else:
                struct.unpack(">II", f.read(8))
                return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    X_train = load_file(os.path.join(path, 'train-images-idx3-ubyte.gz'), True)
    y_train = load_file(os.path.join(path, 'train-labels-idx1-ubyte.gz'), False)
    X_test  = load_file(os.path.join(path, 't10k-images-idx3-ubyte.gz'),  True)
    y_test  = load_file(os.path.join(path, 't10k-labels-idx1-ubyte.gz'),  False)

    mean, std = 0.1307, 0.3081
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    return train_dataset, test_dataset

# --- 2. 模型定义 ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # (N, 8, 14, 14)
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # (N, 16, 7, 7)
            nn.Flatten(),                             # (N, 784)
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. 训练 ---
def train():
    train_dataset, test_dataset = load_mnist_local('data')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    model = CNN().to(device)  # 把模型移到 GPU/CPU
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"参数总量: {sum(p.numel() for p in model.parameters())}")

    loss_history = []
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # 数据也移到同一设备
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 测试集准确率
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            correct += (model(X_batch).argmax(dim=1) == y_batch).sum().item()
    print(f"测试集准确率: {correct / len(test_dataset) * 100:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("模型已保存到 mnist_cnn.pth")

    plt.plot(loss_history)
    plt.title("MNIST Training Loss (PyTorch CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
