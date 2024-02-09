from ultralytics import YOLO
import cv2
import math 
import time
from PIL import Image
import matplotlib as plt
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
actions = ["Looking_forward","Raising_Hand","Reading","Sleeping","Turning_Around","Writting",]
conf_person = 0.75
# model
model = YOLO("yolo-Weights/yolov8n.pt")
model2 = YOLO("runs/classify/train3/weights/best.pt")
while True:
    success, img = cap.read()
    results = model(img, stream=False)

    # coordinates
    for r in results:
        boxes = [x for x in r.boxes if int(x.cls[0]) == 0 and math.ceil((x.conf[0]*100))/100 >= conf_person ]

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            image = Image.fromarray(img)
            cropped_img = image.crop((x1, y1, x2, y2))

            frame_class = model2.predict(cropped_img)[0].probs.data.topk(k=1)




            org = [x1, y1]

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, actions[int(frame_class.indices)], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
