# Hand-coded CNN for MNIST Digit Recognition

A CNN built from scratch using only NumPy, trained on the MNIST handwritten digit dataset.

## Model Architecture

```
Input (N, 1, 28, 28)
  → Conv2D(1→4, 3×3) + ReLU
  → MaxPool2D(2×2)         → (N, 4, 13, 13)
  → Conv2D(4→8, 3×3) + ReLU
  → MaxPool2D(2×2)         → (N, 8, 5, 5)
  → Flatten                → (N, 200)
  → Linear(200→10)
  → Softmax
```

## Requirements

```bash
pip install numpy matplotlib pillow
```

## Dataset

Download the MNIST dataset and place the following files in a `data/` folder:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Available at: http://yann.lecun.com/exdb/mnist/

## Usage

```bash
python handCnn.py
```

Training uses the first 2000 samples (pure Python loops are slow). After training:

1. Loss curve is plotted
2. Accuracy is evaluated on the first 500 test samples (~90%)
3. 10 random test images are shown with predictions
4. A drawing canvas opens — draw a digit and click **Predict**
