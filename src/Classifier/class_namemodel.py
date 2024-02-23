# from keras.models import load_model
# from PIL import Image, ImageOps
# import numpy as np


# def conf():

#     np.set_printoptions(suppress=True)

#     model = load_model("model/pos_model.h5", compile=False)

#     class_names = open("model/labels.txt", "r").readlines()

#     image = Image.open("Classifier/one.jpg").convert("RGB")
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#     image_array = np.asarray(image)
#     normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

#     flattened_image_array = normalized_image_array.flatten()
#     adjusted_input = np.resize(flattened_image_array, (1, 14739))

#     prediction = model.predict(adjusted_input)
#     index = np.argmax(prediction)
#     class_name = class_names[index].strip()  
#     confidence_score = prediction[0][index]
#     return class_name,confidence_score


# # class_name, confidence_score = conf()
# # print(f"Class: {class_name}, Confidence Score: {confidence_score}")

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the complete model, which includes weights
model = load_model("model/pos_model.h5", compile=False)
class_names = open("model/labels.txt", "r").readlines()

def predict(image):
    """
    Predicts the class of the given image.
    
    Args:
    - image: A PIL.Image object in RGB format.
    
    Returns:
    - class_name: The predicted class name.
    - confidence_score: The confidence score of the prediction.
    """
    np.set_printoptions(suppress=True)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    flattened_image_array = normalized_image_array.flatten()
    adjusted_input = np.resize(flattened_image_array, (1, 14739))
    prediction = model.predict(adjusted_input)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  
    confidence_score = prediction[0][index]
    print(class_name)
    return class_name, confidence_score
