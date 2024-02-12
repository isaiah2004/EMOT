import streamlit as st
from ultralytics import YOLO
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
from datetime import datetime


# Load models and initialize settings
device = torch.device('cpu')
model = YOLO("yolov8n.pt",verbose=False).to(device)
model2 = YOLO("model/best.pt",verbose=False).to(device)
actions = ["Sleeping", "Raising_Hand", "Reading", "Looking_forward", "Turning_Around", "Writting"]
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



# df = pd.DataFrame(columns=['Datetime', 'Action'])

collected_data = []

def main():
    st.title("Live Action Detection")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    
    global collected_data 

    def plot():
        # Convert the collected data to a DataFrame and concatenate it with the existing df
        df = pd.read_csv('one.csv')
        # print(df.head(10))
        if not df.empty:
            # st.dataframe(df)
            # Aggregate data for histogram
            action_counts = df['Action'].value_counts()
            st.bar_chart(action_counts)
            st.area_chart(action_counts)
            # pr = df.profile_report()
            # st_profile_report(pr)
        else:
            st.write("No data to display.")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break
        
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the current date and time
        now = datetime.now()

        # Format the date and time
        timestamp = now.strftime("%d:%m:%Y - %H:%M:%S")

        processed_frame, detectedActions = label_cam(frame)

        for action in detectedActions:
            collected_data.append({'Datetime': timestamp, 'Action': action})
            # print(collected_data)
        new_df = pd.DataFrame(collected_data)
        new_df.to_csv('one.csv')
        FRAME_WINDOW.image(processed_frame)
        
        # Using a button to stop might not be responsive enough for real-time applications
        # Instead, the checkbox 'Run' is used to control the loop
    else:
        plot()


    cap.release()


if __name__ == "__main__":
    main()
