from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./src/model/pos_model.h5", compile=False)

# Load the labels
class_names = open("./src/model/labels.txt", "r").readlines()

# Load and preprocess the image
image = Image.open("./src/Classifier/one.jpg").convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Assuming the need to flatten and resize the image data to match the model's input shape
flattened_image_array = normalized_image_array.flatten()
adjusted_input = np.resize(flattened_image_array, (1, 14739))

# Predict using the model
prediction = model.predict(adjusted_input)
index = np.argmax(prediction)
class_name = class_names[index].strip()  # Ensure newline characters are removed
confidence_score = prediction[0][index]

# Print prediction and confidence score
print(f"Class: {class_name}, Confidence Score: {confidence_score}")
