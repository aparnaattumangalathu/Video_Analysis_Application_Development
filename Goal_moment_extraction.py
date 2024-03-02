#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('goal_inceptionv3.h5')


# In[ ]:


import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

# Load the pre-trained InceptionV3 model
from tensorflow.keras.models import load_model
model = load_model('path_to_model_weights.h5')  # Replace with the actual path to your model weights

# Function to preprocess and predict each frame
def predict_frame(frame):
    frame = cv2.resize(frame, (299, 299))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0  # Normalize pixel values
    prediction = model.predict(frame)
    return prediction[0][1]  # Assuming class 1 corresponds to "goal"

# Function to extract goal moments from a video
def extract_goal_moments(video_path, output_path, segment_duration=60, buffer_time=30):
    clip = VideoFileClip(video_path)
    fps = clip.fps

    # Function to check if a frame contains a "goal" moment
    def is_goal_frame(frame):
        prediction = predict_frame(frame)
        return prediction > 0.5

    buffer_frames = int(buffer_time * fps)

    for i, frame in enumerate(clip.iter_frames(fps=fps, dtype='uint8')):
        if is_goal_frame(frame):
            start_time = max(0, (i - buffer_frames) / fps)
            end_time = min(clip.duration, (i + buffer_frames) / fps)

            # Extract and save the segment
            subclip = clip.subclip(start_time, end_time)
            subclip.write_videofile(output_path.format(i), codec="libx264", audio_codec="aac")

            print(f"Goal moment found at {i / fps:.2f} seconds")

# Replace 'path_to_input_video.mp4' and 'path_to_output_segment_{i}.mp4' with your video and output path
input_video_path = 'Football_game.mp4'
output_video_path = 'Highlights\goal_moment{}.mp4'

extract_goal_moments(input_video_path, output_video_path)


# In[ ]:




