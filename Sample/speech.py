import whisper
import time
import torch

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

# Replace 'your_audio_file_path_here.mp3' with the path to your audio file
# and 'transcription_line_timestamps.txt' with your desired output file path
audio_path = 'src/speech/translate.mp3'
output_path = 'transcription_line_timestamps.txt'
transcribe_audio_with_line_timestamps(audio_path, output_path)
