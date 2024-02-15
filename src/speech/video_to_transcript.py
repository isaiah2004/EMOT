import whisper
import time
import torch
from moviepy.editor import VideoFileClip
import os
from datetime import datetime

def transcribe_audio_with_line_timestamps(audio_path, output_path):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  
    model = whisper.load_model("base", device=device)
    start_time = time.time()
    result = model.transcribe(audio_path)

    end_time = time.time()

    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in result['segments']:
            start = segment['start']
            text = segment['text']
            file.write(f"[{start:.2f}] {text}\n")
    
    total_time = end_time - start_time
    print(f"Transcription completed in {total_time:.2f} seconds")

def split_video_and_transcribe(input_video_path, output_directory):
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    audio_output_filename = f"{video_name}_{current_date}_audio.mp3"
    transcription_output_filename = f"{video_name}_{current_date}_transcription.txt"

    start_time = time.time()

    video_clip = VideoFileClip(input_video_path)
    audio_output_path = os.path.join(output_directory, audio_output_filename)
    video_clip.audio.write_audiofile(audio_output_path)
    video_clip.close()

    transcription_output_path = os.path.join(output_directory, transcription_output_filename)
    transcribe_audio_with_line_timestamps(audio_output_path, transcription_output_path)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Video to audio conversion and transcription completed in {total_time:.2f} seconds")

    return audio_output_path, transcription_output_path

input_directory = 'videos'
output_directory = 'output'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
        input_video_path = os.path.join(input_directory, filename)
        audio_path, transcription_path = split_video_and_transcribe(input_video_path, output_directory)
        os.rename(audio_path, os.path.join(output_directory, os.path.basename(audio_path)))
        os.rename(transcription_path, os.path.join(output_directory, os.path.basename(transcription_path)))
