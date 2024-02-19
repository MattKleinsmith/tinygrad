#!/usr/bin/env python3
import numpy as np
from tinygrad import Tensor
from PIL import ImageGrab
from serious_mnist import BigConvNet
import tkinter as tk

class DrawingApp:
  def __init__(self, master, model):
    self.master = master
    self.model = model
    self.canvas = tk.Canvas(master, width=400, height=400, bg="black")
    self.canvas.pack()

    self.label = tk.Label(master, text="Draw a number")
    self.label.pack()

    self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
    self.clear_button.pack()

    self.read_button = tk.Button(master, text="Read", command=self.read_canvas)
    self.read_button.pack()

    self.previous_x = None
    self.previous_y = None

    self.canvas.bind("<B1-Motion>", self.paint)
    self.canvas.bind("<ButtonRelease-1>", self.reset)

  def clear_canvas(self):
    self.canvas.delete("all")

  def paint(self, event):
    paint_color = "white"
    if self.previous_x and self.previous_y:
        self.canvas.create_line(self.previous_x, self.previous_y, event.x, event.y,
                                width=16, fill=paint_color, capstyle=tk.ROUND, smooth=tk.TRUE)
    self.previous_x = event.x
    self.previous_y = event.y

  def reset(self, event):
    self.previous_x = None
    self.previous_y = None

  def read_canvas(self):
    # TODO: Find a GUI library that can handle Retina displays
    x = self.master.winfo_rootx() + 15
    y = self.master.winfo_rooty() + 80
    x1 = x + self.canvas.winfo_width() * 2 - 20
    y1 = y + self.canvas.winfo_height() * 2 - 20

    # TODO: grab() costs 300ms. Find a GUI library that can give pixels
    image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L").resize((28, 28))
    input = Tensor(np.array(image).reshape(-1, 28, 28).astype(np.uint8))
    print(input.numpy())

    output = model.forward(input)
    pred = np.argmax(output.numpy(), axis=-1)[0]
    self.label.config(text=f"{pred}")

if __name__ == "__main__":
  model = BigConvNet()
  model.load("examples/checkpoint968900.safetensors")

  np.set_printoptions(threshold=np.inf, linewidth=np.inf)
  master = tk.Tk()
  DrawingApp(master, model)
  master.mainloop()
