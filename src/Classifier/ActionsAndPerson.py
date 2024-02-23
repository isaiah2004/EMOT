import cv2
import time
from Classifier.label import label_cam

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Action list
actions = ["Sleeping","Raising_Hand","Reading","Looking_forward","Turning_Around","Writting",]
conf_person = 0.75


while True:
    success, img = cap.read()
    currentTime=time.time()
    img, detectedAction= label_cam(img)

    # time.sleep(1)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
