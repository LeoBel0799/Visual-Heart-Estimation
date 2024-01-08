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







"""# Visual Heart Rate Estimation from Facial Video Based on CNN
Bin Huang, Che-Min Chang, Chun-Liang Lin,
Weihai Chen, Chia-Feng Juang and Xingming Wu
"""
main_directory = "home/ubuntu/ecg-fitness_raw-v1.0"
face_detector = dlib.cnn_face_detection_model_v1("/home/ubuntu/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat")


class CustomDataset(Dataset):
    def __init__(self, images_list, dataframe, transform=None):
        self.images_list = images_list
        self.df = dataframe
        self.transform = transform

        # Utilizza direttamente il valore di 'ECG HR' senza calcolare la media dei gruppi
        self.mean_hr_mapping = {i: self.df.loc[i, ' ECG HR'] for i in range(len(self.df))}
        self.image_mapping = {i: img for i, img in enumerate(self.images_list)}
        shared_indices = set(self.mean_hr_mapping.keys()) & set(self.image_mapping.keys())
        shared_indices = sorted(shared_indices)
        self.shared_data = [(self.image_mapping[idx], self.mean_hr_mapping[idx]) for idx in shared_indices]

        # Calcola il massimo e il minimo valore di HR per la normalizzazione
        self.max_hr = max(self.df[' ECG HR'])
        self.min_hr = min(self.df[' ECG HR'])

    def normalize_hr(self, hr):
        # Normalizza HR tra 0 e 1 in base al massimo e al minimo valore di HR nel dataset
        return round((hr - self.min_hr) / (self.max_hr - self.min_hr), 3)

    def __len__(self):
        return len(self.shared_data)

    def __getitem__(self, idx):
        img, mean_hr = self.shared_data[idx]

        if self.transform:
            img = self.transform(img)
        norm_mean_hr = self.normalize_hr(mean_hr)

        return img, norm_mean_hr



def extract_face_region(image, face_box):
    # Extract face bounding box coordinates
    x, y, w, h = face_box.rect.left(), face_box.rect.top(), face_box.rect.width(), face_box.rect.height()

    # Extract the face region
    face_region = image[y:y+h, x:x+w]
    # Resize the face region to 256x192
    feature_image = cv2.resize(face_region, (192, 256))

    # Transpose the image to have dimensions 3x256x192
    feature_image = np.transpose(feature_image, (2, 0, 1))

    return feature_image

def plot_predictions(targets, predictions, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], '--', color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.show()


def process_video(video_path, video_csv_path, face_detector):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    rois_list = []
    resized_rois = []

    # Read the first frame
    ret, frame = cap.read()

    # List to save face regions frame by frame
    face_regions = []

    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Limit the number of frames to analyze
    max_time_to_analyze_seconds = 20  # Adjust the desired time duration in seconds
    max_frames_to_analyze = int(max_time_to_analyze_seconds * frame_rate)
    df = pd.read_csv(video_csv_path)
    # Filtra il DataFrame per ottenere solo le righe entro i primi 12 secondi
    df = df[df['milliseconds'] <= max_time_to_analyze_seconds * 1000]
    #print(df)
    # Create a progress bar
    progress_bar = tqdm(total=min(total_frames, max_frames_to_analyze), position=0, leave=True,
                        desc=f'Processing Frames for {video_path}')

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= max_frames_to_analyze:
            break

        # Detect faces in the frame
        faces = face_detector(frame, 1)

        if faces:
            # Use only the first detected face
            face = faces[0]

            face_region = extract_face_region(frame, face)
            if face_region is not None:
                rois_list.append(face_region)

        progress_bar.update(1)
        frame_count += 1

    progress_bar.close()

    print(f"Video analyzed for {video_path}")
    current_dataset = CustomDataset(rois_list, df)  # Assuming CustomDataset is defined somewhere in your code
    return current_dataset



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
# Instantiate the model
model = CustomModel()

# Print the model architecture
print(model)



main_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
processed_videos = 0
all_datasets = []

for main_dir in main_directories:
    main_dir_path = os.path.join(main_directory, main_dir)

    subdirectories = [d for d in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, d))]
    for sub_dir in subdirectories:
        sub_dir_path = os.path.join(main_dir_path, sub_dir)

        video_files = [f for f in os.listdir(sub_dir_path) if f.endswith("1.avi")]

        for video_file in video_files:
            if processed_videos >= 50:
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
#df.to_csv('/content/drive/MyDrive/Thesis <BELLIZZI>/ecg-fitness_raw-v1.0/dataset-Visual HR Estim by Huang.csv', index=False)


train_size = int(0.8 * len(combined_dataset))
val_size = int(0.1 * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

batch_size = 32
train_loader_F = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_F = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader_F = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definisci la tua funzione di perdita
criterion = nn.MSELoss()

# Definisci l'ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Definisci il numero totale di epoche
total_epochs = 30

# Suddividi le epoche in base al tuo schema di tasso di apprendimento
lr_schedule = {0: 1e-4, 10: 1e-5, 20: 1e-6}
# Definisci il numero di epoche
num_epochs = 20

# Lista per memorizzare la perdita durante l'addestramento e la validazione
train_loss_list = []
val_loss_list = []
rmse_list = []
me_rate_list = []
pearson_correlation_list = []
# Modifica la definizione del tuo DataLoader per convertire i dati in float32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(total_epochs):
    # Cambia il tasso di apprendimento se necessario
    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule[epoch]

    # Addestramento
    model.train()
    train_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        # Converti i dati in float32
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

    # Calcola la perdita media per epoca di addestramento
    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    # Validazione
    model.eval()
    val_loss = 0.0
    predictions = []
    targets_all = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader_F, desc='Validation'):
            # Forward pass
                        # Convert images to float
            images = images.to(dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()


            # Salva le previsioni e i target per il calcolo del RMSE
            predictions.extend(outputs.numpy())
            targets_all.extend(targets.numpy())

    #plot_predictions(targets_all, predictions, f'Validation - Predicted vs Ground Truth')

    # Calcola la perdita media per epoca di validazione
    avg_val_loss = val_loss / len(val_loader_F)
    val_loss_list.append(avg_val_loss)

    # Calcola il RMSE
    rmse = np.sqrt(((np.array(predictions) - np.array(targets_all))**2).mean())
    rmse_list.append(rmse)

    # Calcola la media dell'errore di misura (Me)
    mean_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)))

    # Calcola la deviazione standard dell'errore di misura (SDe)
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



# DataLoader per il set di test
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test
model.eval()
test_loss = 0.0
test_predictions = []
test_targets_all = []

with torch.no_grad():
    for images, targets in tqdm(test_loader, desc='Testing'):
        # Converti le immagini in float
        images = images.to(dtype=torch.float32)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Salva le previsioni e i target per ulteriori analisi
        test_predictions.extend(outputs.numpy())
        test_targets_all.extend(targets.numpy())

# Calcola la perdita media per l'epoca di test
avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# Calcola il RMSE
rmse_test = np.sqrt(((np.array(test_predictions) - np.array(test_targets_all))**2).mean())
print(f'Test RMSE: {rmse_test:.4f}')

plt.plot(test_targets_all, label='Ground Truth')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.title('Test Set: Predicted vs Ground Truth Over Time')
plt.show()


plt.scatter(test_targets_all, test_predictions, label='Predicted vs Ground Truth')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted')
plt.legend()
plt.title('Test Set: Predicted vs Ground Truth')
plt.show()

plt.plot(test_targets_all, label='Ground Truth')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.title('Test Set: Predicted vs Ground Truth Over Time')
plt.show()

torch.save(model, '/home/ubuntu/ecg-fitness_raw-v1.0/CNNRegressor.pth')
# Plot delle perdite durante l'addestramento e la validazione
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Plot della RMSE
plt.plot(rmse_list, label='RMSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) on Validation Set')
plt.show()

# Plot del Mean Absolute Percentage Error
plt.plot(me_rate_list, label='Mean Absolute Percentage Error')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Percentage Error')
plt.title('Mean Absolute Percentage Error on Validation Set')
plt.show()

# Plot di Pearson's Correlation Coefficient
plt.plot(pearson_correlation_list, label="Pearson's Correlation Coefficient")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Pearson's Correlation Coefficient")
plt.title("Pearson's Correlation Coefficient on Validation Set")
plt.show()