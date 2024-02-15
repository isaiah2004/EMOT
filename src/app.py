import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
from label import label_cam

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
