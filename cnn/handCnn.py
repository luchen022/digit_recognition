import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import struct
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# --- 1. 本地数据加载器 (指向你的 data 文件夹) ---
def load_mnist_local(path='data'):
    """
    从本地 data 文件夹加载 MNIST 数据集。
    不再尝试下载，直接读取 .gz 文件。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"错误：找不到文件夹 '{path}'。请确保你把下载的文件放在了当前目录下的 'data' 文件夹中。")

    # 文件名映射 (和你提供的截图一致)
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    def load_file(file_key):
        filename = files[file_key]
        filepath = os.path.join(path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"错误：找不到文件 {filepath}。请检查文件名是否正确。")
            
        print(f"正在读取 {filename} ...")
        with gzip.open(filepath, 'rb') as f:
            if 'images' in file_key:
                # 图像文件头：magic(4), num(4), rows(4), cols(4)
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
                return images.astype(np.float32) / 255.0 # 归一化到 0-1
            else:
                # 标签文件头：magic(4), num(4)
                magic, num = struct.unpack(">II", f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
                return labels

    print("加载训练集...")
    X_train = load_file('train_images')
    y_train_raw = load_file('train_labels')
    
    # 为了演示速度，我们只取前 2000 张图片进行训练
    # 纯 Python 循环处理 60000 张会非常慢（可能需要几小时），2000 张大约几分钟
    limit = 2000 
    X_train = X_train[:limit]
    y_train_raw = y_train_raw[:limit]
    
    # One-hot 编码
    y_train = np.zeros((y_train_raw.size, 10))
    y_train[np.arange(y_train_raw.size), y_train_raw] = 1
    
    print(f"数据加载完成。形状: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train

# --- 2. 神经网络组件 (保持不变) ---
def relu(x): return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
def cross_entropy_loss(y_pred, y_true):
    n_samples = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(n_samples), np.argmax(y_true, axis=1)] + 1e-9)
    return np.sum(log_likelihood) / n_samples

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / k)
        self.b = np.zeros((out_channels, 1))
        self.X_padded = None
        self.dW = None
        self.db = None

    def forward(self, X):
        N, C_in, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        H_out = int((H - K + 2 * P) / S) + 1
        W_out = int((W - K + 2 * P) / S) + 1
        
        if P > 0:
            self.X_padded = np.pad(X, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
        else:
            self.X_padded = X
            
        out = np.zeros((N, self.out_channels, H_out, W_out))
        
        for n in range(N):
            for f in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K
                        slice_input = self.X_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = np.sum(slice_input * self.W[f]) + self.b[f, 0]
        return out

    def backward(self, dout):
        N, F, H_out, W_out = dout.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        
        dX_padded = np.zeros_like(self.X_padded)
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1)
        
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K
                        slice_input = self.X_padded[n, :, h_start:h_end, w_start:w_end]
                        self.dW[f] += dout[n, f, i, j] * slice_input
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += dout[n, f, i, j] * self.W[f]
        
        if P > 0:
            return dX_padded[:, :, P:-P, P:-P]
        return dX_padded

class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride
        self.max_indices = None
        self.cache_shape = None

    def forward(self, X):
        N, C, H, W = X.shape
        S = self.stride
        K = self.size
        H_out = int((H - K) / S) + 1
        W_out = int((W - K) / S) + 1
        out = np.zeros((N, C, H_out, W_out))
        self.cache_shape = (N, C, H, W)
        self.max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=int)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K
                        slice_input = X[n, c, h_start:h_end, w_start:w_end]
                        max_pos = np.unravel_index(np.argmax(slice_input), slice_input.shape)
                        out[n, c, i, j] = slice_input[max_pos]
                        self.max_indices[n, c, i, j] = [h_start + max_pos[0], w_start + max_pos[1]]
        return out

    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        dX = np.zeros(self.cache_shape)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_max, w_max = self.max_indices[n, c, i, j]
                        dX[n, c, h_max, w_max] += dout[n, c, i, j]
        return dX

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.input_cache = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.input_cache = X
        return np.dot(X, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.input_cache.T, dout)
        self.db = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.W.T)

# --- 3. 构建网络 (适应 28x28 输入) ---
# 输入: (N, 1, 28, 28)
# Conv1: 1 -> 4 通道, 3x3 -> (N, 4, 26, 26)
# Pool1: 2x2 -> (N, 4, 13, 13)
# Conv2: 4 -> 8 通道, 3x3 -> (N, 8, 11, 11)
# Pool2: 2x2 -> (N, 8, 5, 5)
# Flatten: 8 * 5 * 5 = 200
# FC: 200 -> 10

conv1 = Conv2D(1, 4, 3, stride=1, padding=0)
pool1 = MaxPool2D(2, 2)
conv2 = Conv2D(4, 8, 3, stride=1, padding=0)
pool2 = MaxPool2D(2, 2)
fc1 = Linear(8 * 5 * 5, 10)

# --- 4. 训练主循环 ---
def train():
    # 使用本地加载函数
    X_train, y_train = load_mnist_local(path='data') 
    
    learning_rate = 0.01
    epochs = 5  
    batch_size = 32
    
    loss_history = []
    
    print("开始训练真实 MNIST 数据 (纯 NumPy 手搓)...")
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        num_batches = len(X_train) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward
            z1 = conv1.forward(X_batch)
            a1 = relu(z1)
            p1 = pool1.forward(a1)
            
            z2 = conv2.forward(p1)
            a2 = relu(z2)
            p2 = pool2.forward(a2)
            
            N = p2.shape[0]
            flat = p2.reshape(N, -1)
            logits = fc1.forward(flat)
            probs = softmax(logits)
            
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss
            
            # Backward
            dlogits = probs - y_batch
            dflat = fc1.backward(dlogits)
            dp2 = dflat.reshape(p2.shape)
            
            da2 = pool2.backward(dp2)
            dz2 = da2 * (z2 > 0) 
            dp1 = conv2.backward(dz2)
            
            da1 = pool1.backward(dp1)
            dz1 = da1 * (z1 > 0)
            dX = conv1.backward(dz1)
            
            # Update (SGD)
            conv1.W -= learning_rate * conv1.dW
            conv1.b -= learning_rate * conv1.db
            conv2.W -= learning_rate * conv2.dW
            conv2.b -= learning_rate * conv2.db
            fc1.W -= learning_rate * fc1.dW
            fc1.b -= learning_rate * fc1.db
            
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("\n训练完成！绘制 Loss 曲线...")
    plt.plot(loss_history)
    plt.title("MNIST Training Loss (Hand-coded CNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 测试集准确率
    evaluate()
    # 可视化预测：随机挑10张测试集图片
    visualize_predictions()
    # 手写画板识别
    draw_and_predict()

def predict(X):
    """对输入图片做前向传播，返回预测类别"""
    z1 = conv1.forward(X)
    a1 = relu(z1)
    p1 = pool1.forward(a1)
    z2 = conv2.forward(p1)
    a2 = relu(z2)
    p2 = pool2.forward(a2)
    flat = p2.reshape(p2.shape[0], -1)
    logits = fc1.forward(flat)
    probs = softmax(logits)
    return np.argmax(probs, axis=1), probs

def evaluate():
    """在测试集上计算准确率"""
    print("\n加载测试集...")
    files = {
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    def load_file(key, path='data'):
        filepath = os.path.join(path, files[key])
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                _, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
                return data.astype(np.float32) / 255.0
            else:
                _, num = struct.unpack(">II", f.read(8))
                return np.frombuffer(f.read(), dtype=np.uint8)

    X_test = load_file('test_images')[:500]  # 取前500张，纯循环太慢
    y_test = load_file('test_labels')[:500]

    preds, _ = predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"测试集准确率 (前500张): {acc * 100:.2f}%")

def visualize_predictions(n=10, path='data'):
    """随机挑 n 张测试集图片，显示图片、真实标签和预测结果"""
    files = {
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    def load_file(key):
        filepath = os.path.join(path, files[key])
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                _, num, rows, cols = struct.unpack(">IIII", f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
                return data.astype(np.float32) / 255.0
            else:
                _, num = struct.unpack(">II", f.read(8))
                return np.frombuffer(f.read(), dtype=np.uint8)

    X_test = load_file('test_images')
    y_test = load_file('test_labels')

    indices = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    preds, probs = predict(X_sample)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    for idx, ax in enumerate(axes):
        ax.imshow(X_sample[idx, 0], cmap='gray')
        color = 'green' if preds[idx] == y_sample[idx] else 'red'
        ax.set_title(f"True:{y_sample[idx]}\nPred:{preds[idx]}", color=color, fontsize=9)
        ax.axis('off')
    plt.suptitle("Green=Correct  Red=Wrong")
    plt.tight_layout()
    plt.show()

def draw_and_predict():
    """弹出画板，让用户手写数字后进行识别"""
    CANVAS_SIZE = 280  # 显示尺寸，内部缩放到28x28

    root = tk.Tk()
    root.title("Handwritten Digit Recognition - Draw a digit")

    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black', cursor='cross')
    canvas.pack()

    label_result = tk.Label(root, text="Draw a digit above, then click Predict", font=('Arial', 14))
    label_result.pack(pady=5)

    # 用 PIL 维护一张图用于预测
    pil_image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=0)
    draw = ImageDraw.Draw(pil_image)

    def paint(event):
        r = 12  # 笔刷半径
        x, y = event.x, event.y
        canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def do_predict():
        # 缩放到 28x28，转成模型需要的格式
        img_28 = pil_image.resize((28, 28), Image.LANCZOS)
        arr = np.array(img_28).astype(np.float32) / 255.0
        arr = arr.reshape(1, 1, 28, 28)
        preds, probs = predict(arr)
        confidence = probs[0, preds[0]] * 100
        label_result.config(text=f"Prediction: {preds[0]}   Confidence: {confidence:.1f}%")

    def clear():
        canvas.delete('all')
        draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        label_result.config(text="Draw a digit above, then click Predict")

    canvas.bind('<B1-Motion>', paint)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)
    tk.Button(btn_frame, text="Predict", font=('Arial', 12), command=do_predict, width=8).pack(side='left', padx=5)
    tk.Button(btn_frame, text="Clear", font=('Arial', 12), command=clear, width=8).pack(side='left', padx=5)

    root.mainloop()

if __name__ == "__main__":
    train()