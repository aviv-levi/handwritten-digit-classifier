import tkinter as tk
import numpy as np
import tensorflow as tf
from digits_predictor import DigitsPredictor

# Constants
CANVAS_SIZE = 28
CELL_SIZE = 10


class Whiteboard:
    def __init__(self, root, digits_predictor: DigitsPredictor):
        self.digits_predictor = digits_predictor
        self.root = root

        self.root.title("Handwritten digit classifier")
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE * CELL_SIZE, height=CANVAS_SIZE * CELL_SIZE)
        self.canvas.pack()
        # setup drawing surface
        self.canvas.bind("<B1-Motion>", self.draw)
        # setup clear-button
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        # setup guess-button
        self.guess_button = tk.Button(self.root, text="Guess", command=self.guess_digit)
        self.guess_button.pack()
        # setup prediction-label
        self.prediction_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.prediction_label.pack()
        # store whiteboard content in numpy 2d array
        self.pixels = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.int)

    def draw(self, event):
        """
            When user drawing, fill pixel and store it
        """
        x, y = event.x // CELL_SIZE, event.y // CELL_SIZE
        self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                     fill="black")
        self.pixels[y, x] = 1

    def clear_canvas(self):
        """
            When user press clear, reset all settings
        """
        self.canvas.delete("all")
        self.pixels = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.int)
        self.prediction_label.config(text="")

    def _pixels_to_input_tensor(self):
        """
            Convert numpy 2d array to 28X28X1 tensor
        """

        # Reshape the pixels array to 28x28x1
        image = np.expand_dims(self.pixels, axis=-1)
        image = np.reshape(image, (1, CANVAS_SIZE, CANVAS_SIZE, 1))
        # Convert to TensorFlow tensor
        return tf.convert_to_tensor(image, dtype=tf.float32)

    def guess_digit(self):
        """
            When user press guess, convert image to tensor, feed model and show prediction
        """
        image_tensor = self._pixels_to_input_tensor()
        prediction_result = self.digits_predictor.predict(image_tensor)
        self.prediction_label.config(text=f"Prediction: {prediction_result}")
