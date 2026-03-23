# PyTorch Deep Learning Study Project

[中文版](README.md)

Classic deep learning models implemented in PyTorch. Each module is self-contained and designed for step-by-step learning.

## Project Structure

```
├── cnn/                        # Convolutional Neural Network — handwritten digit recognition
│   ├── handCnn.py              # Pure NumPy CNN (for understanding the fundamentals)
│   ├── torchCnn.py             # PyTorch CNN (trains and saves the model)
│   ├── predict.py              # Loads the model, drawing canvas for real-time prediction
│   └── data/                   # MNIST dataset
│
└── rnn/                        # Recurrent Neural Network — Chinese sentiment classification
    ├── torchRnn.py             # Basic RNN (demonstrates vanishing gradient problem)
    ├── torchLSTM.py            # LSTM (solves vanishing gradient, better performance)
    ├── lstmPredict.py          # Loads LSTM model, interactive sentiment prediction
    └── ChnSentiCorp_htl_all.csv  # Chinese hotel review dataset (7766 samples)
```

## CNN Module

MNIST handwritten digit recognition, two versions:

### NumPy version (handCnn.py)
Pure NumPy implementation for understanding convolution, pooling, and backpropagation from scratch.
```
Input (N, 1, 28, 28)
  → Conv2D(1→4, 3×3) + ReLU → MaxPool2D(2×2)
  → Conv2D(4→8, 3×3) + ReLU → MaxPool2D(2×2)
  → Flatten → Linear(200→10) → Softmax
```

### PyTorch version (torchCnn.py)
```
Input (N, 1, 28, 28)
  → Conv2D(1→8,  3×3) + ReLU → MaxPool2D(2×2)
  → Conv2D(8→16, 3×3) + ReLU → MaxPool2D(2×2)
  → Flatten → Linear(784→128) + ReLU → Linear(128→10)
```

```bash
python cnn/torchCnn.py   # train, saves to cnn/mnist_cnn.pth
python cnn/predict.py    # open drawing canvas for prediction
```

Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/), place files in `cnn/data/`.

## RNN Module

Chinese hotel review sentiment classification (ChnSentiCorp), binary (positive/negative).

Uses character-level embedding — each Chinese character maps to a learnable vector, no tokenizer needed.

### Basic RNN (torchRnn.py)
Demonstrates the vanishing gradient problem. Unstable training on long sequences, accuracy ~68% (close to majority-class guessing).

### LSTM (torchLSTM.py)
Gating mechanisms solve vanishing gradients, significantly better performance.
```
Input text → character id sequence (length 50)
  → Embedding(vocab_size → 128)
  → LSTM(128 → 128, 2 layers)
  → Linear(128 → 1) → sigmoid → positive/negative
```

```bash
python rnn/torchLSTM.py    # train, saves to rnn/sentiment_lstm.pth
python rnn/lstmPredict.py  # interactive sentiment prediction
```

## Requirements

```bash
pip install torch numpy matplotlib pillow
```
