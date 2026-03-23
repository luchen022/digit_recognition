import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN    = 50
EMBED_DIM  = 128
HIDDEN_DIM = 128

# 模型结构必须和训练时一致
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.rnn(emb)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(1)

# 加载模型和词表
checkpoint = torch.load('sentiment_lstm.pth', map_location=device)
vocab = checkpoint['vocab']

model = SentimentRNN(len(vocab), EMBED_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print("模型加载成功，输入评论进行情感分析（输入 q 退出）\n")

def predict(text):
    ids = [vocab.get(ch, 0) for ch in text[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    x = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    label = "正面 😊" if prob >= 0.5 else "负面 😞"
    print(f"情感：{label}  置信度：{prob * 100:.1f}%")

while True:
    text = input("请输入评论：").strip()
    if text.lower() == 'q':
        break
    if text:
        predict(text)
