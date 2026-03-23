import torch
import torch.nn as nn
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 必须和训练时定义的结构一致
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# 加载保存的权重
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()
print("模型加载成功")

def predict_image(pil_img):
    img = pil_img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0

    # 把数字居中（模仿 MNIST 的处理方式）
    if arr.max() > 0:
        rows = np.any(arr > 0.1, axis=1)
        cols = np.any(arr > 0.1, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = arr[rmin:rmax+1, cmin:cmax+1]
        # 缩放到 20x20 再放到 28x28 中央（MNIST 标准）
        from PIL import Image as PILImage
        cropped_img = PILImage.fromarray((cropped * 255).astype(np.uint8))
        cropped_img = cropped_img.resize((20, 20), PILImage.LANCZOS)
        arr = np.zeros((28, 28), dtype=np.float32)
        arr[4:24, 4:24] = np.array(cropped_img).astype(np.float32) / 255.0

    arr = (arr - 0.1307) / 0.3081
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        pred = probs.argmax().item()
        confidence = probs[0, pred].item() * 100
    return pred, confidence

def main():
    CANVAS_SIZE = 280
    root = tk.Tk()
    root.title("Handwritten Digit Recognition - Draw a digit")

    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black', cursor='cross')
    canvas.pack()

    label_result = tk.Label(root, text="Draw a digit above, then click Predict", font=('Arial', 14))
    label_result.pack(pady=5)

    pil_image = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=0)
    draw = ImageDraw.Draw(pil_image)

    def paint(event):
        r = 12
        x, y = event.x, event.y
        canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def do_predict():
        pred, confidence = predict_image(pil_image)
        label_result.config(text=f"Prediction: {pred}   Confidence: {confidence:.1f}%")

    def clear():
        canvas.delete('all')
        draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        label_result.config(text="Draw a digit above, then click Predict")

    canvas.bind('<B1-Motion>', paint)
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)
    tk.Button(btn_frame, text="Predict", font=('Arial', 12), command=do_predict, width=8).pack(side='left', padx=5)
    tk.Button(btn_frame, text="Clear",   font=('Arial', 12), command=clear,      width=8).pack(side='left', padx=5)
    root.mainloop()

if __name__ == "__main__":
    main()
