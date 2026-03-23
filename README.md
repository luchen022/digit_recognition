# PyTorch 人工智能学习项目

[English Version](README_en.md)

用 PyTorch 实现经典深度学习模型，每个模块独立，适合逐步学习。

## 项目结构

```
├── cnn/                        # 卷积神经网络 —— 手写数字识别
│   ├── handCnn.py              # 纯 NumPy 手搓 CNN（用于理解原理）
│   ├── torchCnn.py             # PyTorch CNN（训练并保存模型）
│   ├── predict.py              # 加载模型，手写画板实时识别
│   └── data/                   # MNIST 数据集
│
└── rnn/                        # 循环神经网络 —— 中文情感分类
    ├── torchRnn.py             # 基础 RNN（演示梯度消失问题）
    ├── torchLSTM.py            # LSTM（解决梯度消失，效果更好）
    ├── lstmPredict.py          # 加载 LSTM 模型，交互式情感预测
    └── ChnSentiCorp_htl_all.csv  # 中文酒店评论数据集（7766 条）
```

## CNN 模块

MNIST 手写数字识别，包含两个版本：

### 手搓版 (handCnn.py)
纯 NumPy 实现，用于理解卷积、池化、反向传播的底层原理。
```
输入 (N, 1, 28, 28)
  → Conv2D(1→4, 3×3) + ReLU → MaxPool2D(2×2)
  → Conv2D(4→8, 3×3) + ReLU → MaxPool2D(2×2)
  → Flatten → Linear(200→10) → Softmax
```

### PyTorch 版 (torchCnn.py)
```
输入 (N, 1, 28, 28)
  → Conv2D(1→8,  3×3) + ReLU → MaxPool2D(2×2)
  → Conv2D(8→16, 3×3) + ReLU → MaxPool2D(2×2)
  → Flatten → Linear(784→128) + ReLU → Linear(128→10)
```

```bash
python cnn/torchCnn.py   # 训练，保存到 cnn/mnist_cnn.pth
python cnn/predict.py    # 打开手写画板识别
```

数据集：[MNIST](http://yann.lecun.com/exdb/mnist/)，下载后放入 `cnn/data/`。

## RNN 模块

中文酒店评论情感分类（ChnSentiCorp），二分类（正面/负面）。

采用字符级 Embedding，每个汉字对应一个可学习向量，无需分词。

### 基础 RNN (torchRnn.py)
用于观察梯度消失现象，长序列下训练不稳定，准确率约 68%（接近猜多数类）。

### LSTM (torchLSTM.py)
通过门控机制解决梯度消失，效果显著提升。
```
输入文本 → 字符 id 序列（长度 50）
  → Embedding(vocab_size → 128)
  → LSTM(128 → 128, 2层)
  → Linear(128 → 1) → sigmoid → 正面/负面
```

```bash
python rnn/torchLSTM.py    # 训练，保存到 rnn/sentiment_lstm.pth
python rnn/lstmPredict.py  # 交互式情感预测
```

## 依赖

```bash
pip install torch numpy matplotlib pillow
```
