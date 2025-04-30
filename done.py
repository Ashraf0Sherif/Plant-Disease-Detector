import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = load_model("my_model.keras")

train_dir = "dataset/train"
class_labels = sorted(os.listdir(train_dir))  

img_path = "test.JPG"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array_expanded = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array_expanded)
predicted_class_index = np.argmax(predictions[0])
predicted_label = class_labels[predicted_class_index]

plt.imshow(img_array)
plt.title(f"Predicted: {predicted_label}", fontsize=14)
plt.axis('off')
plt.show()
