from moviepy.editor import VideoFileClip

def split_video(video_path, audio_output_path, video_output_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)
    
    # Extract audio
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_output_path)
    
    # Extract video
    video_clip.write_videofile(video_output_path, codec='libx264', audio=False)

# Replace 'input_video.mp4' with the path to your input video file
input_video_path = 'D:/Model/EMOT/src/speech/input_file.mp4'

# Replace 'output_audio.mp3' with the desired output path for the audio file
audio_output_path = 'output_audio.mp3'

# Replace 'output_video.mp4' with the desired output path for the video file
video_output_path = 'output_video.mp4'

split_video(input_video_path, audio_output_path, video_output_path)
