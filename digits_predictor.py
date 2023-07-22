import tensorflow as tf
import numpy as np


class DigitsPredictor:
    def __init__(self, model_path: str):
        self.classifier_model = tf.keras.models.load_model(model_path)

    def predict(self, image_tensor: tf.Tensor):
        prediction_probabilities = self.classifier_model.predict(image_tensor)
        prediction_result = np.argmax(prediction_probabilities, axis=1)[0]
        return prediction_result
