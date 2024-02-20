import cv2
import numpy as np
import time
from serious_mnist import BigConvNet
from tinygrad import Tensor

model = BigConvNet()
model.load("examples/checkpoint968900.safetensors")
input_canvas_title = "Draw number. Q: Quit. C: Clear"
output_canvas_title = "Prediction"
width = 400
pen_color = (255, 255, 255)
pen_thickness = 25

def create_blank_canvas(): return np.zeros((width, width, 3), dtype="uint8")

input_canvas = create_blank_canvas()
output_canvas = create_blank_canvas()

last_point = None

def draw(event, x, y, flags, param):
  global last_point
  if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
    if last_point is not None:
      cv2.line(input_canvas, last_point, (x, y), pen_color, pen_thickness)
    last_point = (x, y)
  elif event == cv2.EVENT_LBUTTONUP:
    last_point = None

cv2.namedWindow(input_canvas_title)
cv2.setMouseCallback(input_canvas_title, draw)
cv2.imshow(input_canvas_title, input_canvas)
cv2.imshow(output_canvas_title, output_canvas)
cv2.moveWindow(input_canvas_title, 0, 0)
cv2.moveWindow(output_canvas_title, width, 0)

last_save_time = time.time()
last_prediction = None

try:
  while True:
    cv2.imshow(input_canvas_title, input_canvas)
    cv2.imshow(output_canvas_title, output_canvas)

    if (time.time() - last_save_time < 0.1):
      image = cv2.resize(input_canvas, (28, 28), interpolation=cv2.INTER_AREA)[:, :, 0]
      output = model.forward(Tensor(image))
      prediction = np.argmax(output.numpy(), axis=-1)[0]
      if (last_prediction != prediction):
        output_canvas = create_blank_canvas()
        cv2.putText(output_canvas, str(prediction), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, pen_color, pen_thickness)
      last_prediction = prediction
      last_save_time = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): input_canvas = create_blank_canvas()
finally:
  cv2.destroyAllWindows()
