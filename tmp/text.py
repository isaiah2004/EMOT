import tensorflow as tf
import cv2

model = tf.load_model('./saved_model.h5')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def label_cam(img):
    results = model(img, stream=False)
    detectedAction=[]
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
            color = (255, 100, 100)
            thickness = 2

            cv2.putText(img, actions[int(frame_class.indices)], org, font, fontScale, color, thickness)
            detectedAction.append(actions[int(frame_class.indices)])
    return img, detectedAction


while True:
    success, img = cap.read()
    currentTime=time.time()
    img, detectedAction= label_cam(img)

    # time.sleep(1)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
