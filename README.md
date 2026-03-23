# 手搓 CNN 手写数字识别

[English Version](README_en.md)

仅使用 NumPy 从零实现的 CNN，在 MNIST 手写数字数据集上训练。

## 模型结构

```
输入 (N, 1, 28, 28)
  → Conv2D(1→4, 3×3) + ReLU
  → MaxPool2D(2×2)         → (N, 4, 13, 13)
  → Conv2D(4→8, 3×3) + ReLU
  → MaxPool2D(2×2)         → (N, 8, 5, 5)
  → Flatten                → (N, 200)
  → Linear(200→10)
  → Softmax
```

## 依赖

```bash
pip install numpy matplotlib pillow
```

## 数据集

下载 MNIST 数据集，将以下文件放入 `data/` 文件夹：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

下载地址：http://yann.lecun.com/exdb/mnist/

## 使用方法

```bash
python handCnn.py
```

训练使用前 2000 张样本（纯 Python 循环较慢）。训练完成后：

1. 绘制 Loss 曲线
2. 在前 500 张测试集上评估准确率（约 90%）
3. 随机展示 10 张测试图片及预测结果
4. 弹出画板，在上面手写数字后点击 **Predict** 进行识别
