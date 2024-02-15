import cv2
import os
import datetime

def get_next_filename(directory, prefix="output_", extension=".mkv"):
    """Generates the next file name based on existing files in the specified directory."""
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Format the current datetime for the filename (e.g., "2024-02-15_12-00-00")
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{prefix}{datetime_str}{extension}"
    
    # Check if this filename already exists, add a suffix if it does
    i = 1
    unique_filename = filename
    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{filename}_{i}"
        i += 1
    
    return os.path.join(directory, unique_filename)

# Specify the directory where you want to save the files
save_directory = "videos"

# Define the codec and create VideoWriter object with the unique filename in the specified directory
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
filename = get_next_filename(save_directory)  # Get a unique filename within the specified directory
out = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))

# Open the default camera
cap = cv2.VideoCapture(0)

# Set the capture resolution to 720p if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Write the frame into the file
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

