import os
import sys
import time
import ssl
import random
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats, ndarray
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader, TensorDataset, ConcatDataset
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp
import captum.attr as captum
from facenet_pytorch import *
import neurokit2 as nk
import dlib
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


"""# Implementazione del Paper EVM-CNN: Real-Time Contactless Heart Rate Estimation From Facial Video
"""


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=(kernel_size-1)//2, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.batch_norm(x)
        return F.relu(x)

class CNNBody(nn.Module):
    def __init__(self):
        super(CNNBody, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 5, 1, 2)  # Convolution Layer 1
        self.sep_conv2 = SeparableConvBlock(96, 96, 3, 1)  # Depthwise Convolution Layer 2
        self.sep_conv3 = SeparableConvBlock(96, 96, 1, 1)  # Pointwise Convolution Layer 3
        self.sep_conv4 = SeparableConvBlock(96, 96, 3, 2)  # Depthwise Convolution Layer 4
        self.sep_conv5 = SeparableConvBlock(96, 96, 1, 1)  # Pointwise Convolution Layer 5
        self.sep_conv6 = SeparableConvBlock(96, 128, 3, 2)  # Depthwise Convolution Layer 6
        self.sep_conv7 = SeparableConvBlock(128, 128, 1, 1)  # Pointwise Convolution Layer 7
        self.sep_conv8 = SeparableConvBlock(128, 128, 3, 2)  # Depthwise Convolution Layer 8
        self.sep_conv9 = SeparableConvBlock(128, 128, 1, 1)  # Pointwise Convolution Layer 9
        self.sep_conv10 = SeparableConvBlock(128, 128, 3, 2)  # Depthwise Convolution Layer 10
        self.sep_conv11 = SeparableConvBlock(128, 128, 1, 1)  # Pointwise Convolution Layer 11
        self.avg_pool = nn.AvgPool2d(2, 2)  # Average Pooling Layer
        self.fc1 = nn.Linear(128, 192)  # Fully Connected Layer 1
        self.dropout = nn.Dropout(0.6)  # Dropout Layer
        self.fc2 = nn.Linear(192, 1)  # Fully Connected Layer 2 (Output Layer)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = self.sep_conv4(x)
        x = self.sep_conv5(x)
        x = self.sep_conv6(x)
        x = self.sep_conv7(x)
        x = self.sep_conv8(x)
        x = self.sep_conv9(x)
        x = self.sep_conv10(x)
        x = self.sep_conv11(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNNBody()

class CustomDataset(Dataset):
    def __init__(self, images_list, dataframe, transform=None):
        self.images_list = images_list
        self.df = dataframe
        self.transform = transform

        self.mean_hr_mapping = {i: self.compute_group_mean(i) for i in range(len(self.df) // 31)}
        self.image_mapping = {i: img for i, img in enumerate(self.images_list)}
        shared_indices = set(self.mean_hr_mapping.keys()) & set(self.image_mapping.keys())
        shared_indices = sorted(shared_indices)
        self.shared_data = [(self.image_mapping[idx], self.mean_hr_mapping[idx]) for idx in shared_indices]

        self.max_hr = max(self.df[' ECG HR'])
        self.min_hr = min(self.df[' ECG HR'])

    def compute_group_mean(self, group_idx):
        start_idx = group_idx * 31
        end_idx = min((group_idx + 1) * 31, len(self.df))
        group_data = self.df.loc[start_idx:end_idx - 1, ' ECG HR']
        return round(group_data.mean(), 3)

    def normalize_hr(self, hr):
        return round((hr - self.min_hr) / (self.max_hr - self.min_hr),3)

    def __len__(self):
        return len(self.shared_data)

    def __getitem__(self, idx):
        img, mean_hr = self.shared_data[idx]

        if self.transform:
            img = self.transform(img)

        norm_mean_hr = self.normalize_hr(mean_hr)

        return img, norm_mean_hr

def extract_face_region(image, landmarks):
    landmark_coords = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [2, 14, 40, 41, 46, 47, 50, 52]])

    XLT = landmarks.part(14).x
    YLT = max(landmarks.part(40).y, landmarks.part(41).y, landmarks.part(46).y, landmarks.part(47).y)
    Wrect = landmarks.part(2).x - landmarks.part(14).x
    Hrect = min(landmarks.part(50).y, landmarks.part(52).y) - YLT

    XLT, YLT = int(XLT), int(YLT)
    XRB, YRB = int(XLT + Wrect), int(YLT + Hrect)
    center = (abs(XLT + XRB) / 2, abs(YLT + YRB) / 2)
    size = (abs(XRB - XLT), abs(YRB - YLT))

    if 0 <= XLT < image.shape[1] and 0 <= YLT < image.shape[0] and 0 <= XLT + Wrect <= image.shape[1] and 0 <= YLT + Hrect <= image.shape[0]:
        face_region = cv2.getRectSubPix(image, size, center)

        #rect = cv2.rectangle(image, (int(XLT), int(YLT)), (int(XLT + Wrect), int(YLT + Hrect)), (255, 0, 0), 2)

        # Aggiungi la regione di interesse alla lista
        #rois.append(face_region)

        # Ritorna la copia dell'immagine con il rettangolo disegnato (opzionale)
        return face_region
    else:
        print("Region outside image limits.")
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
    distance = np.sqrt(center[0]**2 + center[1]**2)
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
        intermediate_images = [
            reshape_to_one_column(gaussian_pyramid(frame, Pl))
            for frame in video_frames[:Fps]
        ]
        video_frames = video_frames[Fps:]

        min_rows = min(img.shape[0] for img in intermediate_images)
        intermediate_images = [img[:min_rows, :] for img in intermediate_images]

        C = np.concatenate(intermediate_images, axis=1)

        while C.shape[1] >= Fps:
            # Filtro passa-banda ed estrai le feature
            if C.shape[1] >= 3:
                feature_image = apply_bandpass_filter(C[:, :Fps], Fl, Fh)

                # Rendi la feature image 25x25x3
                feature_image = cv2.resize(feature_image, (25, 25))
                #plt.imshow(feature_image)
                #plt.show()
                feature_image = np.expand_dims(feature_image, axis=-1)
                feature_image = np.concatenate([feature_image] * 3, axis=-1)

                # PyTorch vuole il formato CHW, quindi vado a permutare gli assi
                feature_image = np.transpose(feature_image, (2, 0, 1))

                # Converto in tensore PyTorch
                feature_image_tensor = torch.from_numpy(feature_image).float()


                feature_images.append(feature_image_tensor)

            # Elimino le colonne elaborate
            C = C[:, Fps:]

    return feature_images


Pl = 4
Fps = 30
Fl = 0.75
Fh = 4

def process_video(video_path, video_csv_path, face_detector, landmark_predictor, tracker, Pl, Fps, Fl, Fh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    rois_list = []
    resized_rois = []

    ret, frame = cap.read()

    face_regions = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    max_time_to_analyze_seconds = 15  # Adjust the desired time duration in seconds
    max_frames_to_analyze = int(max_time_to_analyze_seconds * frame_rate)
    df = pd.read_csv(video_csv_path)
    df = df[df['milliseconds'] <= max_time_to_analyze_seconds * 1000]
    progress_bar = tqdm(total=min(total_frames, max_frames_to_analyze), position=0, leave=True,
                        desc=f'Processing Frames for {video_path}')

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= max_frames_to_analyze:
            break

        frame = cv2.resize(frame, (640, 480))

        faces = face_detector(frame, 1)

        if faces and tracker:
            face = faces[0]
            landmarks = landmark_predictor(frame, face.rect)

            face_region = extract_face_region(frame, landmarks)
            if face_region is not None:
                rois_list.append(face_region)

            max_width = max(roi.shape[1] for roi in rois_list)
            max_height = max(roi.shape[0] for roi in rois_list)

            # Vado a fare il resize di tutte le ROI trovate in contemporanea
            resized_rois = [cv2.resize(roi, (max_width, max_height)) for roi in rois_list]

            bbox = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
            tracker.init(frame, bbox)

        if faces and tracker:
            ret, bbox = tracker.update(frame)

            if ret:
                bbox = tuple(map(int, bbox))

                if 0 <= bbox[0] < frame.shape[1] and 0 <= bbox[1] < frame.shape[0] and \
                   0 <= bbox[0] + bbox[2] < frame.shape[1] and 0 <= bbox[1] + bbox[3] < frame.shape[0]:
                    p1 = (bbox[0], bbox[1])
                    p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])

                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        for i in range(68):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 255), -1)

        progress_bar.update(1)
        frame_count += 1

    progress_bar.close()

    print(f"Video analyzed for {video_path}")

    max_width = max(roi.shape[1] for roi in rois_list)
    max_height = max(roi.shape[0] for roi in rois_list)

    resized_rois = [cv2.resize(roi, (max_width, max_height)) for roi in rois_list]

    features_img = extract_features(resized_rois, Pl, Fps, Fl, Fh)

    video_images.extend(features_img)

    print(f"Features extracted for {video_path}")

    current_dataset = CustomDataset(video_images, df)
    return current_dataset



main_directory = "home/ubuntu/ecg-fitness_raw-v1.0"
main_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
face_detector = dlib.cnn_face_detection_model_v1("home/ubuntu/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat")
landmark_predictor = dlib.shape_predictor("home/ubuntu/ecg-fitness_raw-v1.0/shape_predictor_68_face_landmarks_GTX.dat")
tracker = cv2.TrackerCSRT_create()
processed_videos = 0
all_datasets = []

for main_dir in main_directories:
    main_dir_path = os.path.join(main_directory, main_dir)

    subdirectories = [d for d in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, d))]
    for sub_dir in subdirectories:
        sub_dir_path = os.path.join(main_dir_path, sub_dir)

        video_files = [f for f in os.listdir(sub_dir_path) if f.endswith("1.avi")]

        for video_file in video_files:
            if processed_videos >= 30:
               break
            video_images = []
            video_path = os.path.join(sub_dir_path, video_file)
            fin_csv_files = [f for f in os.listdir(sub_dir_path) if f.startswith("fin") and f.endswith(".csv")]

            if len(fin_csv_files) == 1:
                fin_csv_file = fin_csv_files[0]
                video_csv_path = os.path.join(sub_dir_path, fin_csv_file)
            else:
                print(f"Error: No or multiple 'fin' CSV files found in {sub_dir_path}")
                continue

            current_dataset = process_video(video_path,video_csv_path, face_detector,landmark_predictor,tracker, Pl, Fps, Fl, Fh)

            #print(len(current_dataset))
            current_dataset_length = len(current_dataset)
            all_datasets.append(current_dataset)
            print(f"All datasets len: {len(all_datasets)}")
            print(f"CustomDataset created for {video_path} with {current_dataset_length} rows")
            processed_videos += 1
        #cap.release()

combined_dataset = ConcatDataset(all_datasets)

print("Global custom dataset created with length: ",len(combined_dataset))
all_data = []

df = pd.DataFrame(all_data)
df.to_csv('home/ubuntu/ecg-fitness_raw-v1.0/dataset-EVMCNN.csv', index=False)


train_size = int(0.8 * len(combined_dataset))
val_size = int(0.1 * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def plot_predictions(targets, predictions, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], '--', color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.show()



criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

train_loss_list = []
val_loss_list = []
rmse_list = []
me_rate_list = []
pearson_correlation_list = []

for epoch in range(num_epochs):
    # Addestramento
    model.train()
    train_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images = images.float()
        targets = targets.float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    # Validazione
    model.eval()
    val_loss = 0.0
    predictions = []
    targets_all = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # Salva le previsioni e i target per il calcolo del RMSE
            predictions.extend(outputs.numpy())
            targets_all.extend(targets.numpy())

    plot_predictions(targets_all, predictions, f'Validation - Predicted vs Ground Truth')

    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    #  RMSE
    rmse = np.sqrt(((np.array(predictions) - np.array(targets_all))**2).mean())
    rmse_list.append(rmse)

    # Media dell'errore di misura (Me)
    mean_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)))

    # Deviazione standard dell'errore di misura (SDe)
    std_dev_error = np.sqrt(np.mean((np.array(predictions) - np.array(targets_all) - mean_error)**2))

    # Mean Absolute Percentage Error (MeRate)
    mean_absolute_percentage_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)) / np.abs(np.array(targets_all)))
    me_rate_list.append(mean_absolute_percentage_error)

    # Pearson's Correlation Coefficient (ρ)
    mean_ground_truth = np.mean(np.array(targets_all))
    mean_predicted_hr = np.mean(np.array(predictions))
    numerator = np.sum((np.array(targets_all) - mean_ground_truth) * (np.array(predictions) - mean_predicted_hr))
    denominator_ground_truth = np.sum((np.array(targets_all) - mean_ground_truth)**2)
    denominator_predicted_hr = np.sum((np.array(predictions) - mean_predicted_hr)**2)
    pearson_correlation = numerator / np.sqrt(denominator_ground_truth * denominator_predicted_hr)
    pearson_correlation_list.append(pearson_correlation)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, MeRate: {mean_absolute_percentage_error:.4f}, Pearson Correlation: {pearson_correlation:.4f}')

    # Testing
    model.eval()
    test_loss = 0.0
    predictions_test = []
    targets_all_test = []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Save predictions and targets for evaluation
            predictions_test.extend(outputs.numpy())
            targets_all_test.extend(targets.numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_loss_list.append(avg_test_loss)

    rmse_test = np.sqrt(((np.array(predictions_test) - np.array(targets_all_test))**2).mean())
    me_test = np.mean(np.abs(np.array(predictions_test) - np.array(targets_all_test)))
    sde_test = np.sqrt(np.mean((np.array(predictions_test) - np.array(targets_all_test) - me_test)**2))
    me_rate_test = np.mean(np.abs(np.array(predictions_test) - np.array(targets_all_test)) / np.abs(np.array(targets_all_test)))
    pearson_correlation_test = np.corrcoef(np.array(predictions_test), np.array(targets_all_test))[0, 1]

    # Print metrics for the test phase
    print(f'Test Loss: {avg_test_loss:.4f}, RMSE: {rmse_test:.4f}, Mean Error: {me_test:.4f}, SDe: {sde_test:.4f}, MeRate: {me_rate_test:.4f}, Pearson Correlation: {pearson_correlation_test:.4f}')

torch.save(model, 'home/ubuntu/ecg-fitness_raw-v1.0/EVM-CNNRegressor.pth')# Plot delle perdite durante l'addestramento e la validazione


plt.plot(targets_all_test, label='Ground Truth')
plt.plot(predictions_test, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.title('Test Set: Predicted vs Ground Truth Over Time')
plt.show()


plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

plt.plot(rmse_list, label='RMSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) on Validation Set')
plt.show()

plt.plot(me_rate_list, label='Mean Absolute Percentage Error')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Percentage Error')
plt.title('Mean Absolute Percentage Error on Validation Set')
plt.show()

plt.plot(pearson_correlation_list, label="Pearson's Correlation Coefficient")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Pearson's Correlation Coefficient")
plt.title("Pearson's Correlation Coefficient on Validation Set")
plt.show()
