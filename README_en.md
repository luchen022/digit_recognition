# Hand-coded CNN for MNIST Digit Recognition

Two implementations: pure NumPy from scratch, and PyTorch.

## File Structure

```
├── handCnn.py     # Pure NumPy CNN (for learning the fundamentals)
├── torchCnn.py    # PyTorch version (trains and saves the model)
├── predict.py     # Loads the model, drawing canvas for real-time prediction
└── data/          # MNIST dataset (download manually)
```

## Model Architecture

### NumPy version (handCnn.py)
```
Input (N, 1, 28, 28)
  → Conv2D(1→4, 3×3) + ReLU → MaxPool2D(2×2)   → (N, 4, 13, 13)
  → Conv2D(4→8, 3×3) + ReLU → MaxPool2D(2×2)   → (N, 8, 5, 5)
  → Flatten → Linear(200→10) → Softmax
```

### PyTorch version (torchCnn.py)
```
Input (N, 1, 28, 28)
  → Conv2D(1→8,  3×3, padding=1) + ReLU → MaxPool2D(2×2)  → (N, 8, 14, 14)
  → Conv2D(8→16, 3×3, padding=1) + ReLU → MaxPool2D(2×2)  → (N, 16, 7, 7)
  → Flatten → Linear(784→128) + ReLU → Linear(128→10)
```

## Requirements

```bash
pip install numpy matplotlib pillow torch
```

## Dataset

Download the MNIST dataset and place the following files in a `data/` folder:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Available at: http://yann.lecun.com/exdb/mnist/

## Usage

### NumPy version
```bash
python handCnn.py
```
Trains on the first 2000 samples. After training: plots loss curve, shows test predictions, opens drawing canvas.

### PyTorch version
```bash
# Train and save the model
python torchCnn.py

# Load the model and open drawing canvas
python predict.py
```

Training automatically uses GPU if available. Model is saved to `mnist_cnn.pth`.

