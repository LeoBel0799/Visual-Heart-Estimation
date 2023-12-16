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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import dlib
import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset, DataLoader, random_split

"""# CNN + Regression EVMCNN and Huang et al."""


# class CustomModel(nn.Module):
#     def __init__(self):
#         super(CustomModel, self).__init__()
#         self.conv2D_1 = nn.Conv2d(3, 16, kernel_size=1, stride=1)
#         self.conv2D_2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
#         self.conv2D_3 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
#         self.conv2D_4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
#         self.conv2D_5 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
#         self.conv2D_6 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
#
#         self.reshape_conv3D = nn.Flatten()
#         self.lstm_1 = nn.LSTM(512, 128, batch_first=True)
#         self.dropout_1 = nn.Dropout(0.3)
#         self.lstm_2 = nn.LSTM(128, 32, batch_first=True)
#         self.dropout_2 = nn.Dropout(0.3)
#         self.lstm_3 = nn.LSTM(32, 1, batch_first=True)
#         self.reshape_lstm = nn.Flatten()
#         self.dense = nn.Linear(1, 1)
#
#     def forward(self, x):
#         x = self.conv2D_1(x)
#         x = self.conv2D_2(x)
#         x = self.conv2D_3(x)
#         x = self.conv2D_4(x)
#         x = self.conv2D_5(x)
#         x = self.conv2D_6(x)
#
#         x = self.reshape_conv3D(x)
#         x, _ = self.lstm_1(x)
#         x = self.dropout_1(x)
#         x, _ = self.lstm_2(x)
#         x = self.dropout_2(x)
#         x, _ = self.lstm_3(x)
#         x = self.reshape_lstm(x)
#         x = self.dense(x)
#
#         return x
#
# model = CustomModel()
#
# print(model)

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv2D_1 = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.conv2D_2 = nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0)
        self.conv2D_3 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        self.conv2D_4 = nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0)
        self.conv2D_5 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv2D_6 = nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0)
        self.conv2D_7 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv2D_8 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.conv2D_9 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv2D_10 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.conv2D_11 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv2D_12 = nn.Conv2d(256, 384, kernel_size=2, stride=2, padding=0)
        self.conv2D_13 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
        self.conv2D_14 = nn.Conv2d(384, 512, kernel_size=1, stride=2, padding=0)
        self.conv2D_15 = nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0)

        self.reshape_conv3D = nn.Flatten()
        self.lstm_1 = nn.LSTM(512, 128, batch_first=True)
        self.dropout_1 = nn.Dropout(0.3)
        self.lstm_2 = nn.LSTM(128, 32, batch_first=True)
        self.dropout_2 = nn.Dropout(0.3)
        self.lstm_3 = nn.LSTM(32, 1, batch_first=True)
        self.reshape_lstm = nn.Flatten()
        self.dense = nn.Linear(1, 1)

    def forward(self, x):
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.conv2D_3(x)
        x = self.conv2D_4(x)
        x = self.conv2D_5(x)
        x = self.conv2D_6(x)
        x = self.conv2D_7(x)
        x = self.conv2D_8(x)
        x = self.conv2D_9(x)
        x = self.conv2D_10(x)
        x = self.conv2D_11(x)
        x = self.conv2D_12(x)
        x = self.conv2D_13(x)
        x = self.conv2D_14(x)
        x = self.conv2D_15(x)

        x = self.reshape_conv3D(x)
        x, _ = self.lstm_1(x)
        x = self.dropout_1(x)
        x, _ = self.lstm_2(x)
        x = self.dropout_2(x)
        x, _ = self.lstm_3(x)
        x = self.reshape_lstm(x)
        x = self.dense(x)

        return x
model = CustomModel()

print(model)


class CustomDataset(Dataset):
    def __init__(self, images_list, dataframe, transform=None):
        self.images_list = images_list
        self.df = dataframe
        self.transform = transform

        self.mean_hr_mapping = {i: self.df.loc[i, ' ECG HR'] for i in range(len(self.df))}
        self.image_mapping = {i: img for i, img in enumerate(self.images_list)}
        shared_indices = set(self.mean_hr_mapping.keys()) & set(self.image_mapping.keys())
        shared_indices = sorted(shared_indices)
        self.shared_data = [(self.image_mapping[idx], self.mean_hr_mapping[idx]) for idx in shared_indices]

        self.max_hr = max(self.df[' ECG HR'])
        self.min_hr = min(self.df[' ECG HR'])

    def normalize_hr(self, hr):
        return round((hr - self.min_hr) / (self.max_hr - self.min_hr), 3)

    def denormalize_hr(self, norm_hr):
            return round(norm_hr * (self.max_hr - self.min_hr) + self.min_hr, 3)

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

        feature_image = cv2.resize(face_region, (300,200))
        feature_image = np.transpose(feature_image, (2, 0, 1))

        return feature_image
    else:
        print("Region outside image limits.")
        return None

def process_video(video_path, video_csv_path, face_detector):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        rois_list = []
        resized_rois = []

        ret, frame = cap.read()

        # List to save face regions frame by frame
        face_regions = []

        # Calculate the total number of frames in the video
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
            frame = cv2.resize(frame, (300,200))
            faces = face_detector(frame, 1)

            if faces:
                # Use only the first detected face
                face = faces[0]
                landmarks = landmark_predictor(frame, face.rect)

                face_region = extract_face_region(frame, landmarks)
                if face_region is not None:
                    rois_list.append(face_region)
            #for i, face_region in enumerate(rois_list):
                #plt.figure()
                #plt.imshow(face_region)
                #plt.show()
            progress_bar.update(1)
            frame_count += 1

        progress_bar.close()

        print(f"Video analyzed for {video_path}")
        current_dataset = CustomDataset(rois_list, df)  # Assuming CustomDataset is defined somewhere in your code
        return current_dataset
    except Exception as e:
        print(f"Error processing video: {video_path}. Error details: {str(e)}")
        return None  # Returning None to indicate an error
    finally:
        cap.release()



main_directory = "home/ubuntu/ecg-fitness_raw-v1.0"
main_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
face_detector = dlib.cnn_face_detection_model_v1("home/ubuntu/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat")
landmark_predictor = dlib.shape_predictor("home/ubuntu/ecg-fitness_raw-v1.0/shape_predictor_68_face_landmarks_GTX.dat")
processed_videos = 0
all_datasets = []

for main_dir in main_directories:
    main_dir_path = os.path.join(main_directory, main_dir)

    subdirectories = [d for d in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, d))]
    for sub_dir in subdirectories:
        sub_dir_path = os.path.join(main_dir_path, sub_dir)

        video_files = [f for f in os.listdir(sub_dir_path) if f.endswith("1.avi")]

        for video_file in video_files:
            if processed_videos >=50:
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

            current_dataset = process_video(video_path,video_csv_path, face_detector)

            if current_dataset is not None:
                current_dataset_length = len(current_dataset)
                all_datasets.append(current_dataset)
                print(f"All datasets len: {len(all_datasets)}")
                print(f"CustomDataset created for {video_path} with {current_dataset_length} rows")
                processed_videos += 1
            else:
                print(f"Skipping video {video_path} due to processing error.")


combined_dataset = ConcatDataset(all_datasets)

print("Global custom dataset created with length: ",len(combined_dataset))
all_data = []

df = pd.DataFrame(all_data)
df.to_csv('home/ubuntu/ecg-fitness_raw-v1.0/dataset-CNNRegMix.csv', index=False)



train_size = int(0.8 * len(combined_dataset))
val_size = int(0.1 * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

total_samples_in_train_loader = len(train_loader.dataset)
print(f"Total number of samples in train_loader: {total_samples_in_train_loader}")


total_samples_in_val_loader = len(val_loader.dataset)
print(f"Total number of samples in test_loader: {total_samples_in_val_loader}")

total_samples_in_test_loader = len(test_loader.dataset)
print(f"Total number of samples in test_loader: {total_samples_in_test_loader}")


# Assuming each video lasts for 8 seconds and you have FPS frames per second
video_duration_seconds = 30
fps = 30  # Change this to your actual FPS
# Assuming combined_dataset is an instance of ConcatDataset
all_hr_values = []

for dataset in combined_dataset.datasets:
    # Access df attribute from each CustomDataset in ConcatDataset
    current_df = dataset.df

    total_frames = len(current_df)
    frames_per_video = video_duration_seconds * fps

    for start_frame in range(0, total_frames, frames_per_video):
        end_frame = start_frame + frames_per_video
        video_hr_values = current_df.loc[start_frame:end_frame - 1, ' ECG HR']
        all_hr_values.extend(video_hr_values)

# Plot the concatenated HR values
plt.plot(all_hr_values, label='HR Values')
plt.xlabel('Frame Index')
plt.ylabel('HR Values')
plt.title('Concatenated HR Values from All Videos')
plt.legend()
plt.show()

# Assuming each video lasts for 8 seconds and you have 30 frames per second
video_duration_seconds = 30
fps = 30  # Change this to your actual FPS

# Assuming combined_dataset is an instance of ConcatDataset
video_index_to_plot = 0  # Change this to the video index you want to plot
dataset_to_plot = combined_dataset.datasets[video_index_to_plot]

# Access df attribute from the CustomDataset you want to plot
current_df = dataset_to_plot.df

# Get the frame index and HR values for the chosen video
frame_indices = range(len(current_df))
hr_values = current_df[' ECG HR']

# Plot the HR values
plt.plot(frame_indices, hr_values, label='HR Values')
plt.xlabel('Frame Index')
plt.ylabel('HR Values')
plt.title(f'HR Values for Video Index {video_index_to_plot}')
plt.legend()
plt.show()

# Stampa elementi di train_loader
for batch_idx, (images, targets) in enumerate(train_loader):
    print(f"Train Batch {batch_idx + 1} - Images shape: {images.shape}, Targets shape: {targets.shape}")

# Estrai le immagini e i target dal primo batch di train_loader
first_batch_images = images.numpy()
first_batch_targets = targets.numpy()

# Visualizza le prime 4 immagini del batch
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(np.transpose(first_batch_images[i], (1, 2, 0)))  # Trasponi per adattarlo a imshow
    plt.title(f'Target: {first_batch_targets[i]}')

plt.show()



model = model.cuda()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

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
        images = images.float().cuda()
        targets = targets.float().cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calcola la perdita media per epoca di addestramento
    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    # Validazione
    model.eval()
    val_loss = 0.0
    predictions = []
    targets_all = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            # Converti i dati in float32
            images = images.float().cuda()
            targets = targets.float().cuda()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())

    #plot_predictions(targets_all, predictions, f'Validation - Predicted vs Ground Truth')

    # Calcola la perdita media per epoca di validazione
    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    #  RMSE
    rmse = np.sqrt(((np.array(predictions) - np.array(targets_all))**2).mean())
    rmse_list.append(rmse)

    # Media dell'errore di misura (Me)
    mean_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)))

    # Deviazione standard dell'errore di misura (SDe)
    std_dev_error = np.sqrt(np.mean((np.array(predictions) - np.array(targets_all) - mean_error)**2))

    # Calcola il Mean Absolute Percentage Error (MeRate)
    mean_absolute_percentage_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)) / np.abs(np.array(targets_all)))
    me_rate_list.append(mean_absolute_percentage_error)

    # Calcola Pearson's Correlation Coefficient (ρ)
    mean_ground_truth = np.mean(np.array(targets_all))
    mean_predicted_hr = np.mean(np.array(predictions))
    numerator = np.sum((np.array(targets_all) - mean_ground_truth) * (np.array(predictions) - mean_predicted_hr))
    denominator_ground_truth = np.sum((np.array(targets_all) - mean_ground_truth)**2)
    denominator_predicted_hr = np.sum((np.array(predictions) - mean_predicted_hr)**2)
    pearson_correlation = numerator / np.sqrt(denominator_ground_truth * denominator_predicted_hr)
    pearson_correlation_list.append(pearson_correlation)

    # Stampa le perdite e le metriche per ogni epoca
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, MeRate: {mean_absolute_percentage_error:.4f}, Pearson Correlation: {pearson_correlation:.4f}')


# Test
model.eval()
test_loss = 0.0
test_predictions = []
test_targets_all = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc='Testing'):
        # Converti le immagini in float
        images = images.float().cuda()
        targets = targets.float().cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Salva le previsioni e i target per ulteriori analisi
        test_predictions.extend(outputs.cpu().numpy())
        test_targets_all.extend(targets.cpu().numpy())

# Calcola la perdita media per l'epoca di test
avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

#  RMSE
rmse_test = np.sqrt(((np.array(test_predictions) - np.array(test_targets_all))**2).mean())
print(f'Test RMSE: {rmse_test:.4f}')

plt.plot(test_targets_all, label='Ground Truth')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.title('Test Set: Predicted vs Ground Truth Over Time')
plt.show()


torch.save(model, 'home/ubuntu/ecg-fitness_raw-v1.0/CNNRegressorMIXED.pth')
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Plot RMSE
plt.plot(rmse_list, label='RMSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) on Validation Set')
plt.show()

# Plot  Mean Absolute Percentage Error
plt.plot(me_rate_list, label='Mean Absolute Percentage Error')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Percentage Error')
plt.title('Mean Absolute Percentage Error on Validation Set')
plt.show()

# Plot Pearson's Correlation Coefficient
plt.plot(pearson_correlation_list, label="Pearson's Correlation Coefficient")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Pearson's Correlation Coefficient")
plt.title("Pearson's Correlation Coefficient on Validation Set")
plt.show()