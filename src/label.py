from ultralytics import YOLO
import cv2
import math
from PIL import Image
import torch



# Load models and initialize settings
device = torch.device('cuda')
model = YOLO("yolov8n.pt",verbose=False).to(device)
model2 = YOLO("D:/Model/EMOT/model/best.pt",verbose=False).to(device)
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
            frame_class = model2.predict(cropped_img)[0].probs.data.topk(k=1)
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 100, 100)
            thickness = 2
            cv2.putText(img, actions[int(frame_class.indices)], org, font, fontScale, color, thickness)
            detectedAction.append(actions[int(frame_class.indices)])
    return img, detectedAction
