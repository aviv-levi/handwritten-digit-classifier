import tkinter as tk
from whiteboard import Whiteboard
from digits_predictor import DigitsPredictor

if __name__ == "__main__":
    digits_predictor = DigitsPredictor(model_path='model/digit_classifier_model')

    root = tk.Tk()
    app = Whiteboard(root, digits_predictor)
    root.mainloop()
