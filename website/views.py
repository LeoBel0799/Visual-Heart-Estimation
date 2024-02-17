from flask import Response, Blueprint, render_template, send_file, jsonify
from flask_login import login_required, current_user
from flask import Blueprint, flash, render_template, request, redirect, url_for
import torch
import cv2
from flask_opencv_streamer.streamer import Streamer
from flask import Response, stream_with_context
import tqdm
from werkzeug.utils import secure_filename
import os
import base64
from facenet_pytorch import *
import neurokit2 as nk
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import dlib
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import optuna
import dlib
import os
import cv2
import cv2
import torch
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset, DataLoader, random_split
import math
from torch.optim import Adam
from einops import rearrange
from einops.layers.torch import Rearrange
from einops import repeat
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from .models import db,Video
from sqlalchemy import desc
from .models import VideoRecord

views = Blueprint('views',__name__)

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)

ALLOWED_EXT = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT






min_value = 53.0
max_value = 127.0

# Load the saved ViT model and set it to evaluation mode
model = torch.jit.load(r'C:/Users/user/Desktop/Visual-Heart-Estimation/VIT_jit_cpu.pt', map_location=torch.device('cpu'))
model.eval()

face_cascade = cv2.CascadeClassifier(r'C:/Users/user/Desktop/Visual-Heart-Estimation/haarcascade_frontalface_default.xml')

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
])


def preprocess_frame(frame):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    processed_frame = transform(frame_pil).unsqueeze(0)
    return processed_frame

@views.route('/index')
def index():
    return render_template('index.html')




@views.route('/predict2', methods=['POST'])
def predict2():
    try:
        # Get the video file from the request
        video_file = request.files['file']

        # Save the uploaded video in the local directory
        video_path = 'uploaded_video.mp4'
        video_file.save(video_path)

        # Save the uploaded video in the database
        video_record = VideoRecord(file_path=video_path)
        db.session.add(video_record)
        db.session.commit()

        # Open the video
        video_capture = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = 'output_video.mp4'
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process each frame in the video
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]

                # Preprocess the face
                processed_face = preprocess_frame(face_roi)

                # Make a prediction using the ViT model
                with torch.no_grad():
                    output = model(processed_face)

                # Convert the normalized output to the original scale
                normalized_prediction = output.item()
                original_prediction = normalized_prediction * (max_value - min_value) + min_value
                original_prediction = round(original_prediction, 2)

                # Draw the prediction on the frame
                cv2.putText(frame, f'Predicted BPM: {original_prediction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            # Display the frame in the browser in real-time
            cv2.imshow('Video Prediction', frame)
            cv2.waitKey(1)

            # Write the frame to the output video file
            video_writer.write(frame)

            frame_count += 1

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display results every 10 seconds
            if frame_count % (fps * 5) == 0:
                print(f'Prediction at frame {frame_count}: {original_prediction}')

            # Release resources and destroy window if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        # Release resources
        video_capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # Return the path to the output video file as HTML
        return render_template('index.html', output_video_path=output_video_path)

    except Exception as e:
        return jsonify({'error': str(e)})


from flask import request, render_template, jsonify
from .models import VideoRecord
import cv2
import torch


@views.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the video file from the request
        video_file = request.files['file']

        # Save the uploaded video in the local directory
        video_path = 'uploaded_video.mp4'
        video_file.save(video_path)

        # Save the uploaded video in the database
        video_record = VideoRecord(file_path=video_path)
        db.session.add(video_record)
        db.session.commit()

        # Open the video
        video_capture = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = 'output_video.mp4'
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Define the delay between frames in milliseconds (increase for slower playback)
        delay_between_frames = 50  # 50 milliseconds delay between frames

        # Process each frame in the video
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]

                # Preprocess the face
                processed_face = preprocess_frame(face_roi)

                # Make a prediction using the ViT model
                with torch.no_grad():
                    output = model(processed_face)

                # Convert the normalized output to the original scale
                normalized_prediction = output.item()
                original_prediction = normalized_prediction * (max_value - min_value) + min_value
                original_prediction = round(original_prediction, 2)

                # Draw the prediction on the frame
                cv2.putText(frame, f'Predicted BPM: {original_prediction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Write the frame to the output video file
            video_writer.write(frame)

            frame_count += 1

            # Display results every 10 seconds
            if frame_count % (fps * 5) == 0:
                print(f'Prediction at frame {frame_count}: {original_prediction}')

            # Add delay to control playback speed
            cv2.waitKey(delay_between_frames)

        # Release resources
        video_capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # Return the path to the output video file as HTML
        return render_template('index.html', output_video_path=output_video_path)

    except Exception as e:
        return jsonify({'error': str(e)})
