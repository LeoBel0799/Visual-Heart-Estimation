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


face_detector = 'mmod_human_face_detector.dat'
landmark_predictor = 'shape_predictor_68_face_landmarks_GTX.dat'
file_path = 'min_max_values_vit.txt'
model_path = '/home/ah/Desktop/Flask-BPM-Project/VIT_jit_cpu.pt'

# face_detector = dlib.cnn_face_detection_model_v1(face_detector)
landmark_predictor = dlib.shape_predictor(landmark_predictor)

Pl = 4
Fps = 30
Fl = 0.75
Fh = 4

with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    key, value = line.strip().split(': ')
    if key == 'min_mean_hr':
        min_mean_hr = float(value)
    elif key == 'max_mean_hr':
        max_mean_hr = float(value)



def denormalize(y, max_v, min_v):
    final_value = y * (max_v - min_v) + min_v
    return final_value


def extract_face_region(image, landmarks):
    XLT = landmarks.part(14).x
    YLT = max(landmarks.part(40).y, landmarks.part(41).y, landmarks.part(46).y, landmarks.part(47).y)
    Wrect = landmarks.part(2).x - landmarks.part(14).x
    Hrect = min(landmarks.part(50).y, landmarks.part(52).y) - YLT

    XLT, YLT = int(XLT), int(YLT)
    XRB, YRB = int(XLT + Wrect), int(YLT + Hrect)
    center = (abs(XLT + XRB) / 2, abs(YLT + YRB) / 2)
    size = (abs(XRB - XLT), abs(YRB - YLT))

    if 0 <= XLT < image.shape[1] and 0 <= YLT < image.shape[0] and 0 <= XLT + Wrect <= image.shape[
        1] and 0 <= YLT + Hrect <= image.shape[0]:
        face_region = cv2.getRectSubPix(image, size, center)

        if face_region is None:
            return None

        elif len(face_region.shape) == 3:
            feature_image = cv2.resize(face_region, (200, 200))
            feature_image = feature_image / 255.0
            feature_image = np.moveaxis(feature_image, -1, 0)
        else:
            feature_image = cv2.resize(face_region, (200, 200))
            # feature_image_res = np.expand_dims(feature_image, axis=0)

        return feature_image
    else:
        print("[!] - W: Region outside image limits.")
        return None


def gaussian_pyramid(frame, level):
    for _ in range(level - 1):
        frame = cv2.pyrDown(frame)
    return frame


def reshape_to_one_column(frame):
    return frame.reshape(-1, 1)


def ideal_bandpass_filter(shape, Fl, Fh):
    rows, cols = np.indices(shape)
    center = (rows - shape[0] // 2, cols - shape[1] // 2)
    distance = np.sqrt(center[0] ** 2 + center[1] ** 2)
    mask = (distance >= Fl) & (distance <= Fh)
    return mask.astype(float)


def apply_bandpass_filter(image, Fl, Fh):
    Mf = np.fft.fftshift(np.fft.fft2(image))

    mask = ideal_bandpass_filter(image.shape, Fl, Fh)
    N = Mf * mask
    Ni = np.fft.ifft2(np.fft.ifftshift(N)).real

    return Ni

def extract_features(video_frames, Pl, Fps, Fl, Fh):
     feature_images = []

     while len(video_frames) >= Fps:
         # Process Fps frames
         intermediate_images = [
             reshape_to_one_column(gaussian_pyramid(frame, Pl))
             for frame in video_frames[:Fps]
         ]
         video_frames = video_frames[Fps:]

         # Regola la dimensione lungo l'asse 0
         min_rows = min(img.shape[0] for img in intermediate_images)
         intermediate_images = [img[:min_rows, :] for img in intermediate_images]

         # Concatenate columns
         C = np.concatenate(intermediate_images, axis=1)

         while C.shape[1] >= Fps:
             # Applica il filtro passa-banda ed estrai le feature
             if C.shape[1] >= 3:
                 feature_image = apply_bandpass_filter(C[:, :Fps], Fl, Fh)

                 # Rendi la feature image 25x25x3
                 feature_image = cv2.resize(feature_image, (40, 40))
                 # plt.imshow(feature_image)
                 # plt.show()
                 feature_image = np.expand_dims(feature_image, axis=-1)
                 feature_image = np.concatenate([feature_image] * 3, axis=-1)

                 # PyTorch vuole il formato CHW, quindi permuta gli assi
                 feature_image = np.transpose(feature_image, (2, 0, 1))

                 # Converti in tensore PyTorch
                 feature_image_tensor = torch.from_numpy(feature_image).float()

                 # Salva il tensore delle feature
                 feature_images.append(feature_image_tensor)

             # Rimuovi le colonne elaborate
             C = C[:, Fps:]

     return feature_images

def draw_annotations(frame, face, predicted_value):
    x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    bpm_text = f'BPM: {predicted_value:.2f}'
    cv2.putText(frame, bpm_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame



@views.route('/choose_video', methods=['POST'])
def choose_video():

    if 'video' not in request.files:
        flash('No video file found!', category='error')
        return redirect(url_for('views.home'))

    video = request.files['video']

    if video.filename == " ":
        flash('No video file selected!', category='error')
        return redirect(url_for('views.home'))

    if video and allowed_file(video.filename):
        video_data = video.read()
        new_video = Video(video_data=video_data, path=video.filename)
        db.session.add(new_video)
        db.session.commit()
        flash('Video verified ad uploaded successfully!', category='success')

    else:
        flash('File in the wrong format. Only mp4 allowed!', category='error')

    return redirect(url_for('views.home'))


def index():
    last_video = Video.query.order_by(Video.id.desc()).first()
    return last_video.id if last_video else None


#model_load = torch.jit.load(model_path)

model_load = torch.jit.load(model_path, map_location=torch.device('cpu'))

model_load.eval()

def index():
    last_video = Video.query.order_by(Video.id.desc()).first()
    return last_video.id if last_video else None


class VideoCamera(object):
    def __init__(self):
        self.video = None

    def __del__(self):
        if self.video:
            self.video.release()

    def get_video_from_db(self):
        video_data = Video.query.get(index())
        if video_data:
            self.video = cv2.VideoCapture(video_data.path)
            return self.video

    def get_frame(self):
        if self.video:
            _, image = self.video.read()
            _, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return b'', None

def gen(camera):
    pass
    # frame_count = 0
    # while True:
    #     frame = camera.get_frame()
    #     frame_count += 1
    #     if frame_count % 40 == 0:
    #         faces = face_detector(frame, 1)
    #         if faces:
    #             face = faces[0]
    #             print("Faccia trovata")
    #             landmarks = landmark_predictor(frame, face.rect)
    #             face_region = extract_face_region(frame, landmarks)
    #             print("Regione viso estratta")
    #             if face_region is not None:
    #                 img = extract_features(face_region, Pl, 30, Fl, Fh)
    #                 print("Regione viso estratta")
    #                 with torch.no_grad():
    #                     output = model_load(img)

    #                 predicted_value = output.item()
    #                 pred_denorm = denormalize(predicted_value, max_mean_hr, min_mean_hr)
    #                 annotated_frame = draw_annotations(frame.copy(), face, pred_denorm)

    #                 yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
    #                        cv2.imencode('.jpg', annotated_frame)[1].tobytes() + b'\r\n\r\n')
    #         else:
    #             frame_count += 1
    #     else:
    #         frame_count += 1


@views.route('/get_video', methods=['GET'])
def get_video():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


##########################################################################################################################################

import cv2
import torch
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from torchvision import transforms

min_value = 53.0  # Replace with your actual min value
max_value = 127.0  # Replace with your actual max value

# Load the saved ViT model and set it to evaluation mode
model = torch.jit.load('/home/ah/Desktop/Flask-BPM-Project/VIT_jit_cpu.pt', map_location=torch.device('cpu'))
model.eval()

face_cascade = cv2.CascadeClassifier('/home/ah/Desktop/Flask-BPM-Project/haarcascade_frontalface_default.xml')

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


# from flask import request, render_template, jsonify
# from .models import VideoRecord
# import cv2
# import torch


# @views.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the video file from the request
#         video_file = request.files['file']

#         # Save the uploaded video in the local directory
#         video_path = 'uploaded_video.mp4'
#         video_file.save(video_path)

#         # Save the uploaded video in the database
#         video_record = VideoRecord(file_path=video_path)
#         db.session.add(video_record)
#         db.session.commit()

#         # Open the video
#         video_capture = cv2.VideoCapture(video_path)

#         # Get video properties
#         fps = int(video_capture.get(cv2.CAP_PROP_FPS))
#         width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Define codec and create a VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         output_video_path = 'output_video.mp4'
#         video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#         # Define the delay between frames in milliseconds (increase for slower playback)
#         delay_between_frames = 50  # 50 milliseconds delay between frames

#         # Process each frame in the video
#         frame_count = 0
#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 break

#             # Detect faces in the frame
#             faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#             # Process each detected face
#             for (x, y, w, h) in faces:
#                 face_roi = frame[y:y + h, x:x + w]

#                 # Preprocess the face
#                 processed_face = preprocess_frame(face_roi)

#                 # Make a prediction using the ViT model
#                 with torch.no_grad():
#                     output = model(processed_face)

#                 # Convert the normalized output to the original scale
#                 normalized_prediction = output.item()
#                 original_prediction = normalized_prediction * (max_value - min_value) + min_value
#                 original_prediction = round(original_prediction, 2)

#                 # Draw the prediction on the frame
#                 cv2.putText(frame, f'Predicted BPM: {original_prediction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#             # Write the frame to the output video file
#             video_writer.write(frame)

#             frame_count += 1

#             # Display results every 10 seconds
#             if frame_count % (fps * 5) == 0:
#                 print(f'Prediction at frame {frame_count}: {original_prediction}')

#             # Add delay to control playback speed
#             cv2.waitKey(delay_between_frames)

#         # Release resources
#         video_capture.release()
#         video_writer.release()
#         cv2.destroyAllWindows()

#         # Return the path to the output video file as HTML
#         return render_template('index.html', output_video_path=output_video_path)

#     except Exception as e:
#         return jsonify({'error': str(e)})

from flask import render_template, jsonify, request
import cv2
import cv2 as cv
from website.models import VideoRecord, db  
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np


@views.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize the webcam
        video_capture = cv2.VideoCapture(0)

        # Get video properties
        fps = 30  # 30 fps for webcam
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Counter for video file name
        video_counter = 1
        output_video_path = f'output_video_{video_counter}.mp4'

        # Check if the output video file already exists, increment the counter if needed
        while os.path.exists(output_video_path):
            video_counter += 1
            output_video_path = f'output_video_{video_counter}.mp4'

        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Define the delay between frames in milliseconds (increase for slower playback)
        delay_between_frames = 50  # 50 milliseconds delay between frames

        # Set the end time for prediction (30 seconds)
        end_time = cv2.getTickCount() + int(30 * cv2.getTickFrequency())

        # Process frames from the webcam
        while cv2.waitKey(1) & 0xFF != ord('q'):  # Press 'q' to exit
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
                cv2.putText(frame, f'Predicted BPM: {original_prediction}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Write the frame to the output video file
            video_writer.write(frame)

            # Display the frame
            cv2.imshow('Webcam Prediction', frame)

            # Break if the prediction duration is over
            if cv2.getTickCount() >= end_time:
                break

        # Release resources
        video_capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # Save the predicted video path to the database (if needed)
        video_record = VideoRecord(file_path=output_video_path)
        db.session.add(video_record)
        db.session.commit()

        # Return the path to the output video file as HTML
        return render_template('index.html', output_video_path=output_video_path)

    except Exception as e:
        return jsonify({'error': str(e)})
