# 手搓 CNN 手写数字识别

[English Version](README_en.md)

包含两个实现版本：纯 NumPy 手搓版和 PyTorch 版。

## 文件结构

```
├── handCnn.py     # 纯 NumPy 手搓 CNN（用于学习原理）
├── torchCnn.py    # PyTorch 版（训练并保存模型）
├── predict.py     # 加载模型，手写画板实时识别
└── data/          # MNIST 数据集（需自行下载）
```

## 模型结构

### 手搓版 (handCnn.py)
```
输入 (N, 1, 28, 28)
  → Conv2D(1→4, 3×3) + ReLU → MaxPool2D(2×2)   → (N, 4, 13, 13)
  → Conv2D(4→8, 3×3) + ReLU → MaxPool2D(2×2)   → (N, 8, 5, 5)
  → Flatten → Linear(200→10) → Softmax
```

### PyTorch 版 (torchCnn.py)
```
输入 (N, 1, 28, 28)
  → Conv2D(1→8,  3×3, padding=1) + ReLU → MaxPool2D(2×2)  → (N, 8, 14, 14)
  → Conv2D(8→16, 3×3, padding=1) + ReLU → MaxPool2D(2×2)  → (N, 16, 7, 7)
  → Flatten → Linear(784→128) + ReLU → Linear(128→10)
```

## 依赖

```bash
pip install numpy matplotlib pillow torch
```

## 数据集

下载 MNIST 数据集，将以下文件放入 `data/` 文件夹：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

下载地址：http://yann.lecun.com/exdb/mnist/

## 使用方法

### 手搓版
```bash
python handCnn.py
```
训练使用前 2000 张样本。训练完成后绘制 Loss 曲线、展示测试集预测结果、弹出手写画板。

### PyTorch 版
```bash
# 训练并保存模型
python torchCnn.py

# 加载模型，打开手写画板识别
python predict.py
```

训练自动检测并使用 GPU（如有），完成后保存模型到 `mnist_cnn.pth`。
