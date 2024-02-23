from ultralytics import YOLO
import cv2
import math
from PIL import Image
import torch
# Import the predict function from class_namemodel
from Classifier.class_namemodel import predict
 
# Load models and initialize settings
device = torch.device('cuda')
model = YOLO("yolov8n.pt", verbose=False).to(device)
# model2 variable is no longer needed as we directly call predict now
actions = ["Sleeping", "Raising_Hand", "Reading", "Looking_forward", "Turning_Around", "Writing"]
conf_person = 0.75

def label_cam(img):
    results = model(img, stream=False)
    detectedAction = []
    for r in results:
        boxes = [x for x in r.boxes if int(x.cls[0]) == 0 and math.ceil((x.conf[0]*100))/100 >= conf_person]
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            image = Image.fromarray(img)
            cropped_img = image.crop((x1, y1, x2, y2))
            # Use the predict function with the cropped image
            class_name , confidence_score = predict(cropped_img)
            # Find the index of the class_name in actions to get the correct action
            # action_index = actions.index(class_name) if class_name in actions else -1
            
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 100, 100)
            thickness = 2
            cv2.putText(img, class_name, org, font, fontScale, color, thickness)
            detectedAction.append(class_name)
    return img, detectedAction
