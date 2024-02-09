from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

# Explicitly define the device as CPU
device = torch.device('cpu')

ActionOnlineStatus=False
# Load a pretrained YOLO model (recommended for training)
# Create an instance of the model architecture
# model = YOLO('./model/best.pt').to(device)
model = YOLO().to(device)

modal = torch.load('./model/actionRecognitionModel.pt')

# Load the state dictionary into the model
# state_dict = torch.load('./model/actionRecognitionModel.pt')
# model.load_state_dict(state_dict)
# model.eval()  # Set the model to evaluation mode

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize the image to the size your model expects
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])

# image_path = './src/APIserver/read10_rgb_2_frame31.png'
image_path = './src/APIserver/example.jpg'

image = Image.open(image_path)
transforemed_image = transform(image).unsqueeze(0).to(device)  # Add a batch dimension

# Predict action
def predict_action(img):
    with torch.no_grad():  # No need to track gradients
        return model(img)

# Print the model's prediction
print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOI')

predictions=predict_action(transforemed_image)


# for prediction in predictions:
print(predictions[0])

draw = ImageDraw.Draw(image)



# Check if the predictions contain bounding boxes
for prediction in predictions:
    if prediction.boxes is not None:
        for (i,box,conf) in zip(prediction.boxes.cls,prediction.boxes.xyxy,prediction.boxes.conf):
            print(int(i),box,float(conf))
            if(int(i) == 0 and float(conf)>.5):
                # The box format is expected to be [x1, y1, x2, y2]
                print(i,box,conf)
                bbox = tuple(box.cpu().numpy()) # Convert to NumPy array if not already
                draw.rectangle(bbox, outline="red", width=2)





# Display the image with bounding boxes
image.show()


# # Assuming predictions are tensors and of the format [x1, y1, x2, y2, confidence, class_id]
# if predictions is not None:
#     for pred in predictions[0]:
#         # bbox = pred[:4].tolist()  # Convert bounding box coordinates to list
#         bbox = box[:4].cpu().numpy()  # Convert to NumPy array if not already
#         draw.rectangle(bbox, outline="red", width=2)

# # Display the image with bounding boxes
# image.show()