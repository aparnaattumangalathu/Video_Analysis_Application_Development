from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from typing import List, Dict
from keras.models import Model

app = Flask(__name__)

# Load emotion recognition model
emotion_model = load_model("models/emotion_recognition_model.keras")

# Load football model
football_model = load_model("models/football_video_model.keras")

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Constants for scenario identification
SEGMENT_LENGTH = 60  # seconds
TIME_BUFFER = 30  # seconds

# Define emotion labels
emotion_labels = {3: 'Happy', 4: 'Sad', 5: 'Surprise'}

# Define football scenario labels
football_labels = {0: 'Goal', 1: 'Happy', 2: 'Loss'}


# Function to extract video segments based on frame count
def extract_video_segments(video_path, num_segments):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Capture FPS from the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segments = []
    if total_frames > 0:
        segment_length = int(total_frames / num_segments)
        for i in range(num_segments):
            start_frame = i * segment_length
            end_frame = min((i + 1) * segment_length, total_frames)
            segment_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                segment_frames.append(frame)
            segments.append(segment_frames)

    cap.release()
    return segments


# Function to detect faces in a frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces


# Function to crop faces from a frame based on detected face coordinates
def crop_faces(frame, faces):
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y + h, x:x + w]
        cropped_faces.append(cropped_face)
    return cropped_faces


def preprocess_frame_for_emotion(frame):
    # Extract the first frame from the segment
    frame = frame[0]

    # Resize the frame to the input size expected by the emotion recognition model
    resized_frame = cv2.resize(frame, (48, 48))

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to be in the range [0, 1]
    normalized_frame = gray_frame / 255.0

    expanded_frame = np.expand_dims(normalized_frame, axis=0)

    return expanded_frame


def preprocess_frame_for_football(frame):
    # Extract the first frame from the segment
    frame = frame[0]

    # Resize the frame to the input size expected by the football model
    resized_frame = cv2.resize(frame, (224, 224))

    return resized_frame


def analyze_emotions(segments):
    emotions = []
    for segment_frames in segments:
        segment_emotion = []
        try:
            # Use only the first frame for emotion analysis
            frame = segment_frames[0]

            faces = detect_faces(frame)
            facial_images = crop_faces(frame, faces)

            if facial_images:
                # Assuming single face per frame 
                segment_emotion.append(predict_emotion(facial_images[0]))
            else:
                segment_emotion.append('No faces detected')
        except Exception as e:
            segment_emotion.append(f'Error: {e}')
        emotions.append(segment_emotion)  # Append the segment emotions list
    return emotions
def predict_emotion(image):
    emotion_logits = emotion_model.predict(np.expand_dims(image, axis=0))
    emotion_index = np.argmax(emotion_logits)
    return emotion_labels[emotion_index]


def identify_football_scenarios(video_segments: List[np.ndarray],
                                  emotion_model: Model,
                                  football_model: Model,
                                  fps: float) -> Dict[str, List[float]]:
    scenarios = {}

    for segment_frames in video_segments:
        fps = len(segment_frames) / SEGMENT_LENGTH  # Calculate FPS based on segment length
        for frame in segment_frames:
            timestamp = frame / fps  # Calculate timestamp based on frame index and FPS

            # Preprocess frame for emotion model
            emotion_image = preprocess_frame_for_emotion(frame)

            # Predict emotion
            emotion_prediction = emotion_model.predict(np.expand_dims(emotion_image, axis=0))
            predicted_emotion_index = np.argmax(emotion_prediction)
            predicted_emotion = emotion_labels[predicted_emotion_index]

            # Preprocess frame for football model
            scenario_image = preprocess_frame_for_football(frame)

            # Predict scenario using football model
            scenario_prediction = football_model.predict(np.expand_dims(scenario_image, axis=0))

            # Convert football model output to scenario label
            predicted_scenario = football_labels.get(scenario_prediction[0], "No scenario detected")

            # Combine emotion and football model predictions
            if predicted_scenario == "Goal" and predicted_emotion in ("Happy", "Surprise"):
                scenario = "Goal"  # Prioritize "Goal" if detected and emotion is relevant
            elif predicted_scenario == "Happy" and predicted_emotion == "Happy":
                scenario = "Happy"  # Set scenario to "Happy" if both models predict "Happy"
            elif predicted_scenario == "Sad" and predicted_emotion == "Sad":
                scenario = "Loss"  # Set scenario to "Sad" if both models predict "Sad"
            else:
                scenario = predicted_scenario  # Use football model prediction otherwise

            if scenario in scenarios:
                scenarios[scenario].append(timestamp)  # Append timestamp to existing scenario list
            else:
                scenarios[scenario] = [timestamp]  # Create a new list for the scenario

    return scenarios



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])


def analyze():
    if request.method == 'POST' and 'video' in request.files:
        try:
            video_file = request.files['video']
            video_path = 'uploads/' + video_file.filename
            video_file.save(video_path)

            # Extract video segments (assuming fixed segment length)
            num_segments = int(video_file.content_length / (5 * 1024 * 1024))  # Approximate for 5MP frames (adjust based on video size)
            video_segments = extract_video_segments(video_path, num_segments)

            # Calculate FPS
            total_frames = sum(len(segment) for segment in video_segments)
            duration_seconds = total_frames / fps
            fps = total_frames / duration_seconds

            # Identify football scenarios and timestamps
            scenarios = identify_football_scenarios(video_segments, emotion_model, football_model, fps)

            # Analyze emotions in segments (using only first frame per segment)
            emotions = analyze_emotions([segment[0] for segment in video_segments])

            # Create video segments for each scenario
            for scenario, timestamps in scenarios.items():
                for timestamp in timestamps:
                    # Calculate pre- and post-scenario buffer in frames based on FPS and TIME_BUFFER
                    buffer_frames = int(TIME_BUFFER * fps)
                    buffer_start = int(max(0, timestamp * fps - buffer_frames // 2))
                    buffer_end = int(min(len(video_segments[0]), timestamp * fps + buffer_frames // 2))

                    # Open the original video
                    cap = cv2.VideoCapture(video_path)

                    # Extract video frames based on buffer and scenario frame
                    scenario_frames = []
                    for frame_index in range(buffer_start, buffer_end):
                        ret, frame = cap.read(frame_index)
                        if not ret:
                            break
                        scenario_frames.append(frame)
                    cap.release()

                    # Check and handle video segment exceeding 60 seconds (optional)
                    max_segment_frames = int(fps * 60)  # Maximum frames for 60 seconds
                    if len(scenario_frames) > max_segment_frames:
                        # Implement logic to trim or split the segment (e.g., create multiple videos)
                        pass

                    # Create a new video segment for the scenario with timestamp information
                    scenario_video_name = f"{scenario}_{timestamp:.2f}.mp4"  # Include scenario and timestamp
                    out_cap = cv2.VideoWriter(scenario_video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (scenario_frames[0].shape[1], scenario_frames[0].shape[0]))
                    for frame in scenario_frames:
                        out_cap.write(frame)
                    out_cap.release()
                    
            return render_template('results.html', scenarios=scenarios, emotions=emotions)

        except Exception as e:
            return render_template('error.html', error_message=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
