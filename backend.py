from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io 

app = Flask(__name__)
model = load_model("my_model.keras")

train_dir = "dataset/train"
class_labels = sorted(os.listdir(train_dir))

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img_file = request.files["file"]

    img_bytes = img_file.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]

    return jsonify({"class": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
