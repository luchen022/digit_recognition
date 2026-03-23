import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# --- 超参数 ---
MAX_LEN    = 50    # 每条评论截断/填充到的最大字符数
EMBED_DIM  = 128    # 每个字符的向量维度
HIDDEN_DIM = 128   # LSTM 隐藏层维度
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 2e-3

# --- 1. 数据加载与词表构建 ---
def load_data(path='ChnSentiCorp_htl_all.csv'):
    texts, labels = [], []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['review'].strip():
                texts.append(row['review'].strip())
                labels.append(int(row['label']))
    return texts, labels

def build_vocab(texts):
    """字符级词表：每个唯一字符分配一个 id，0 保留给 <PAD>"""
    chars = set()
    for t in texts:
        chars.update(t)
    vocab = {ch: i + 1 for i, ch in enumerate(sorted(chars))}
    vocab['<PAD>'] = 0
    return vocab

def encode(text, vocab, max_len):
    """文本 → 定长整数序列（截断或 PAD 填充）"""
    ids = [vocab.get(ch, 0) for ch in text[:max_len]]
    ids += [0] * (max_len - len(ids))   # 右侧 PAD
    return ids

# --- 2. Dataset ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [
            (torch.tensor(encode(t, vocab, max_len), dtype=torch.long),
             torch.tensor(label, dtype=torch.float32))
            for t, label in zip(texts, labels)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 3. 模型 ---
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        # Embedding：词表大小 × 向量维度 的可学习查找表
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM：相比 RNN 多了一个 cell state，能更好地记住长距离依赖
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                           num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        # 取最后时间步的隐藏状态做二分类
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.rnn(emb)       # h_n: (num_layers, batch, hidden_dim)，c_n 丢掉
        out = self.dropout(h_n[-1])       # 取最后一层: (batch, hidden_dim)
        return self.fc(out).squeeze(1)    # (batch,)  logits

# --- 4. 训练 ---
def train():
    texts, labels = load_data('ChnSentiCorp_htl_all.csv')

    # 打乱后按 8:2 划分训练/测试集
    data = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data, test_data = data[:split], data[split:]

    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    print(f"词表大小: {vocab_size}")

    train_texts, train_labels = zip(*train_data)
    test_texts,  test_labels  = zip(*test_data)

    train_loader = DataLoader(
        SentimentDataset(train_texts, train_labels, vocab, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        SentimentDataset(test_texts, test_labels, vocab, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentRNN(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"参数总量: {sum(p.numel() for p in model.parameters())}")

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # 测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = (torch.sigmoid(model(x_batch)) >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    print(f"测试集准确率: {correct / total * 100:.2f}%")

    torch.save({'model_state': model.state_dict(), 'vocab': vocab}, 'sentiment_lstm.pth')
    print("模型已保存到 sentiment_lstm.pth")

    plt.plot(loss_history)
    plt.title("ChnSentiCorp Training Loss (LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
