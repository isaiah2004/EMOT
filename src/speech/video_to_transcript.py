import whisper
import time
import torch
from moviepy.editor import VideoFileClip
import os
from datetime import datetime

def transcribe_audio_with_line_timestamps(audio_path, output_path):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Inform the user about the device in use

    # Load the pre-trained Whisper model with GPU support if available
    model = whisper.load_model("base", device=device)

    # Start time for processing measurement
    start_time = time.time()

    # Transcribe the audio, ensuring the model uses the specified device
    result = model.transcribe(audio_path)

    # End time for processing measurement
    end_time = time.time()

    # Write the transcription with line timestamps to a file
    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in result['segments']:
            # Assuming each segment is a line, prepend each line with its start timestamp
            start = segment['start']
            text = segment['text']
            # Format the line with its start timestamp
            file.write(f"[{start:.2f}] {text}\n")
    
    # Calculate and print total time taken for transcription
    total_time = end_time - start_time
    print(f"Transcription completed in {total_time:.2f} seconds")

def split_video_and_transcribe(input_video_path, output_directory):
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    # Get current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Generate dynamic file names
    audio_output_filename = f"{video_name}_{current_date}_audio.mp3"
    transcription_output_filename = f"{video_name}_{current_date}_transcription.txt"

    # Start time for processing measurement
    start_time = time.time()

    # Split the video to extract audio
    video_clip = VideoFileClip(input_video_path)
    audio_output_path = os.path.join(output_directory, audio_output_filename)
    video_clip.audio.write_audiofile(audio_output_path)
    video_clip.close()

    # Transcribe the audio with line timestamps
    transcription_output_path = os.path.join(output_directory, transcription_output_filename)
    transcribe_audio_with_line_timestamps(audio_output_path, transcription_output_path)

    # Delete the temporary audio file
    # os.remove(audio_output_path)

    # End time for processing measurement
    end_time = time.time()

    # Calculate and print total time taken for both processes
    total_time = end_time - start_time
    print(f"Video to audio conversion and transcription completed in {total_time:.2f} seconds")

    # Return the paths of the extracted audio and transcription files
    return audio_output_path, transcription_output_path

# Define the input directory containing video files
input_directory = './videos/'

# Define the output directory
output_directory = 'output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over each file in the input directory
for filename in os.listdir(input_directory):
    # Check if the file is a video file
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
        # Construct the full path to the video file
        input_video_path = os.path.join(input_directory, filename)
        
        # Call the function to split video and transcribe audio
        audio_path, transcription_path = split_video_and_transcribe(input_video_path, output_directory)

        # Move the extracted audio file to the output directory
        os.rename(audio_path, os.path.join(output_directory, os.path.basename(audio_path)))

        # Move the transcription file to the output directory
        os.rename(transcription_path, os.path.join(output_directory, os.path.basename(transcription_path)))
