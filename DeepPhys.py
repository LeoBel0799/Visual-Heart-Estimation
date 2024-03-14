import os
import sys
import time
import ssl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, r2_score
import random
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats, ndarray
import cv2
import optuna
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

from facenet_pytorch import *
import neurokit2 as nk
import dlib
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")



class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class DeepPhys(nn.Module):
    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, img_size=36):
        super(DeepPhys, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                       bias=True)
        self.motion_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('[x] - W: Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, x, params=None):
        d1 = torch.tanh(self.motion_conv1(x))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(x))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out

model = DeepPhys()
print(model)



class CustomDataset:
    def __init__(self, images_list, ecg_hr_values_list, transform=None):
        self.images_list = images_list
        self.ecg_hr_values_list = ecg_hr_values_list
        self.transform = transform

    def __getitem__(self, index):

        img = self.images_list[index]

        avg_hr = self.ecg_hr_values_list[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, avg_hr

    def __len__(self):
        self.num_images = len(self.images_list)
        return self.num_images



class CustomDatasetNormalized(Dataset):
    def __init__(self, img_and_label):
        self.img_and_label = img_and_label

    def __len__(self):
        return len(self.img_and_label)

    def __getitem__(self, idx):
        img, hr_norm, hr_original = self.img_and_label[idx]
        return img, hr_norm, hr_original

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

                 # Rendi la feature image
                 feature_image = cv2.resize(feature_image, (36, 36))
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


Pl = 4
Fps = 10
Fl = 0.75
Fh = 4


def process_video(video_path, video_csv_path, face_detector, landmark_predictor, tracker, Pl, Fps, Fl, Fh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] - W: Error opening video file: {video_path}")
        return

    rois_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    max_time_to_analyze_seconds = 60
    max_frames_to_analyze = int(max_time_to_analyze_seconds * frame_rate)
    df = pd.read_csv(video_csv_path, usecols=['milliseconds', ' ECG HR'])
    df = df[df['milliseconds'] <= max_time_to_analyze_seconds * 1000]
    progress_bar = tqdm(total=min(total_frames, max_frames_to_analyze), position=0, leave=True,
                        desc=f'Processing Frames for {video_path}')

    frame_count = 0
    frame_c = 0
    ecg_frame_count = 0
    video_images = []
    ecg_hr_values = []
    count = 0
    #print("count reset")
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames_to_analyze:
            break

        if frame_count % 10 == 0:
            faces = face_detector(frame, 1)

            if faces:
                count += 1
                #print(f"count: {count}, DataFrame length: {len(df)}")
                ecg_hr_value = df.iloc[count, df.columns.get_loc(" ECG HR")]
                ecg_hr_values.append(ecg_hr_value)
                ecg_frame_count += 1
                frame_c += 1

        if not faces:
            frame_count += 1
            continue

        face = faces[0]
        landmarks = landmark_predictor(frame, face.rect)

        face_region = extract_face_region(frame, landmarks)
        if face_region is not None:
            rois_list.append(face_region)
            bbox = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
            tracker.init(frame, bbox)
        else:
            print("[!] - W: Issues with face region in current frame. Skipping...")
            frame_count += 1
            continue

        progress_bar.update(1)
        frame_count += 1
    progress_bar.close()

    if ecg_frame_count == frame_c:

        features_img = extract_features(rois_list, Pl, Fps, Fl, Fh)
        video_images.extend(features_img)
        print(f"[+] - OK: Features extracted for {video_path}")
        current_dataset = CustomDataset(video_images, ecg_hr_values)
        max_hr = max(current_dataset.ecg_hr_values_list)
        min_hr = min(current_dataset.ecg_hr_values_list)

        print(f"[INFO] - Max value: {max_hr}, Min value: {min_hr}")

        return current_dataset, max_hr, min_hr
    else:
        print("[!] - W: Mismatch in frame counts. Unable to create CustomDataset.")
        return None


def process_and_create_dataset(main_directory, video_to_process):
    main_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    face_detector = dlib.cnn_face_detection_model_v1(
        "/home/ubuntu/data/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat")
    landmark_predictor = dlib.shape_predictor(
        "/home/ubuntu/data/ecg-fitness_raw-v1.0/shape_predictor_68_face_landmarks_GTX.dat")
    processed_videos = 0
    all_datasets = []
    global_max_value = float('-inf')
    global_min_value = float('inf')
    for main_dir in main_directories:
        main_dir_path = os.path.join(main_directory, main_dir)

        subdirectories = [d for d in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, d))]
        for sub_dir in subdirectories:
            sub_dir_path = os.path.join(main_dir_path, sub_dir)

            video_files = [f for f in os.listdir(sub_dir_path) if f.endswith("1.avi")]

            for video_file in video_files:
                if processed_videos >= video_to_process:
                    break

                video_path = os.path.join(sub_dir_path, video_file)
                fin_csv_files = [f for f in os.listdir(sub_dir_path) if f.startswith("ecg") and f.endswith(".csv")]
                for csv_file in fin_csv_files:
                    file_path = os.path.join(sub_dir_path, csv_file)
                    try:
                        print(f"Try to read: {file_path}")
                        df = pd.read_csv(file_path)
                        df[' ECG HR'] = np.where((df[' ECG HR'] < 30), np.random.randint(88, 93, size=len(df)),
                                                 df[' ECG HR'])
                        df[' ECG HR'] = df[' ECG HR'].abs()
                        df = df[df[' ECG HR'] >= 0]
                    except pd.errors.EmptyDataError:
                        print(f"[!] - W: '{file_path}' is empty. Skipping...")
                        continue

                if len(fin_csv_files) == 1:
                    fin_csv_file = fin_csv_files[0]
                    video_csv_path = os.path.join(sub_dir_path, fin_csv_file)
                    tracker = cv2.TrackerGOTURN_create()
                    current_dataset, max_val, min_val = process_video(video_path, video_csv_path, face_detector, landmark_predictor,
                                                    tracker,
                                                    Pl, Fps, Fl, Fh)
                    global_max_value = max(global_max_value, max_val)
                    global_min_value = min(global_min_value, min_val)
                else:
                    print(f"[x] - E: No or multiple 'fin' CSV files found in {sub_dir_path}")
                    continue

                if current_dataset is not None:
                    current_dataset_length = len(current_dataset)
                    all_datasets.append(current_dataset)
                    print(f"[INFO] - All datasets len: {len(all_datasets)}")
                    print(f"[INFO] - CustomDataset created for {video_path} with {current_dataset_length} rows")
                    processed_videos += 1
                else:
                    print(f"[!] - W: Skipping video {video_path} due to processing error.")

    combined_dataset = ConcatDataset(all_datasets)

    print("[INFO] - Global custom dataset created with length: ", len(combined_dataset))

    return combined_dataset, global_min_value, global_max_value


def normalize(custom_dataset, normalized_dataset, max_mean_hr, min_mean_hr):

    for i in range(len(custom_dataset)):
        img, mean_hr = custom_dataset[i]
        norm_mean_hr = (mean_hr - min_mean_hr) / (max_mean_hr - min_mean_hr)
        normalized_dataset.append([img, norm_mean_hr, mean_hr])
    return CustomDatasetNormalized(normalized_dataset)


def denormalize(y, max_v, min_v):
    final_value = y * (max_v - min_v) + min_v
    return final_value


main_directory = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/video"
video_to_process = 99
dataset_path = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Data.pth"
#----------UNCOMMENT TO PROCESS VIDEO AND TRAIN DATA------------------------------
final_dataset, min_val, max_val = process_and_create_dataset(main_directory, video_to_process)
norm_dataset = []
normalized_dataset = normalize(final_dataset, norm_dataset, max_val, min_val)
torch.save(normalized_dataset, dataset_path)
print("[INFO] -Global custom dataset saved!")
# ---------------------------------------------------------------------------------

loaded_dataset = torch.load(dataset_path)
print("[INFO] - Global custom dataset loaded...!")

save_file_path = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/min_max_values_dataset.txt"

loaded_dataset = torch.load(dataset_path)
print("[INFO] - Global custom dataset loaded...!")


with open(save_file_path, 'w') as file:
        file.write(f'min_hr: {min_val}\n')
        file.write(f'max_hr: {max_val}\n')

with open(save_file_path, 'r') as file:
       lines = file.readlines()

min_val_line = lines[0].strip().split(': ')
max_val_line = lines[1].strip().split(': ')
min_val = float(min_val_line[1])
max_val = float(max_val_line[1])

print(f'min_val: {min_val}')
print(f'max_val: {max_val}')

train_size = int(0.7 * len(loaded_dataset))
val_size = int(0.1 * len(loaded_dataset))
test_size = len(loaded_dataset) - train_size - val_size



train_dataset, val_dataset, test_dataset = random_split(loaded_dataset, [train_size, val_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=6)

total_samples_in_train_loader = len(train_loader.dataset)
print(f"[INFO] - Total number of samples in train_loader: {total_samples_in_train_loader}")

total_samples_in_val_loader = len(val_loader.dataset)
print(f"[INFO] - Total number of samples in val_loader: {total_samples_in_val_loader}")

total_samples_in_test_loader = len(test_loader.dataset)
print(f"[INFO] - Total number of samples in test_loader: {total_samples_in_test_loader}")

criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device,f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",)


def objective(trial, train, val):
    LR = trial.suggest_float('LR', 1e-6, 1e-3, log=True)
    WD = trial.suggest_float('WD', 1e-8, 1e-4, log=True)
    num_epochs = trial.suggest_int('num_epochs', 90, 100)
    print(f"[INFO] - Trying parameters: LR={LR}, WD={WD}, num of epochs={num_epochs}")
    model = DeepPhys()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_rmse = float('inf')
    no_improvement_count = 0

    train_loss_list = []
    val_loss_list = []
    rmse_list_val = []
    best_val_rmse = float('inf')
    no_improvement_count = 0
    model.to(device)
    normalized_value_list_validation = []
    denormalized_values_list_validation = []
    train_loss_list = []

    averaged_losses = []
    averaged_val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            images, targets_norm, targets_original = batch
            images, targets_norm = images.to(device), targets_norm.to(device)
            images = images.to(torch.float32)
            targets_norm = targets_norm.to(torch.float32)
            outputs = model(images)
            loss = criterion(outputs, targets_norm)
            rmse_loss = torch.sqrt(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs} loss: {rmse_loss:.2f}")

        train_loss_list.append(rmse_loss.item())

        if (epoch + 1) % 10 == 0:
            avg_loss = sum(train_loss_list[-10:]) / 10
            averaged_losses.append(avg_loss)
            print(f"Averaged train losses: {averaged_losses}")

        model.eval()
        predictions_dict = {}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images, targets_norm, targets_original = batch
                images, targets_norm = images.to(device), targets_norm.to(device)

                images = images.to(torch.float32)
                targets_norm = targets_norm.to(torch.float32)

                outputs = model(images)
                for i in range(len(outputs)):
                    predictions_dict[i] = {
                        'target_norm': targets_norm[i].item(),
                        'prediction': outputs[i].item()
                    }

        mse_sum = 0
        for i in predictions_dict:
            prediction = predictions_dict[i]['prediction']
            target_norm = predictions_dict[i]['target_norm']
            mse_sum += (prediction - target_norm) ** 2

        mse = mse_sum / len(predictions_dict)
        rmse_validation = np.sqrt(mse)
        rmse_list_val.append(rmse_validation)

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation RMSE: {rmse_validation:.4f}")

        if (epoch + 1) % 10 == 0:
            avg_val_loss = sum(rmse_list_val[-10:]) / 10
            averaged_val_losses.append(avg_val_loss)
            min_val_loss = min(averaged_val_losses)
            print(f"Averaged validation losses: {averaged_val_losses}")
            print(f"Minimum validation loss: {min_val_loss}")

            if len(averaged_val_losses) >= 5 and all(val > min_val_loss for val in averaged_val_losses[-5:]):
                print("Breaking loop as the last 5 values are greater than the minimum.")
                break

    return min_val_loss


study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=15)
best_params = study.best_params
print("[INFO] - Best Hyperparameters:", best_params)
best_numepochs = best_params['num_epochs']
best_lr = best_params['LR']
best_wd= best_params['WD']


LR = best_lr
WD = best_wd
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
num_epochs = 5000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)


best_val_rmse = float('inf')
no_improvement_count = 0

train_loss_list = []
val_loss_list = []
rmse_list_val = []
best_val_rmse = float('inf')
no_improvement_count = 0
model.to(device)
normalized_value_list_validation = []
denormalized_values_list_validation = []
train_loss_list = []



averaged_losses = []
averaged_val_losses = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, targets_norm, targets_original = batch
        images, targets_norm = images.to(device), targets_norm.to(device)
        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)
        outputs = model(images)
        loss = criterion(outputs, targets_norm)
        rmse_loss = torch.sqrt(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} loss: {rmse_loss:.2f}")


    train_loss_list.append(rmse_loss.item())

    if (epoch + 1) % 10 == 0:
        avg_loss = sum(train_loss_list[-10:]) / 10
        averaged_losses.append(avg_loss)
        print(f"Averaged train losses: {averaged_losses}")

    model.eval()
    predictions_dict = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images, targets_norm, targets_original = batch
            images, targets_norm = images.to(device), targets_norm.to(device)

            images = images.to(torch.float32)
            targets_norm = targets_norm.to(torch.float32)

            outputs = model(images)
            for i in range(len(outputs)):
                predictions_dict[i] = {
                    'target_norm': targets_norm[i].item(),
                    'prediction': outputs[i].item()
                }


    mse_sum = 0
    for i in predictions_dict:
        prediction = predictions_dict[i]['prediction']
        target_norm = predictions_dict[i]['target_norm']
        mse_sum += (prediction - target_norm) ** 2

    mse = mse_sum / len(predictions_dict)
    rmse_validation = np.sqrt(mse)
    rmse_list_val.append(rmse_validation)

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation RMSE: {rmse_validation:.4f}")

    if (epoch + 1) % 10 == 0:
        avg_val_loss = sum(rmse_list_val[-10:]) / 10
        averaged_val_losses.append(avg_val_loss)
        min_val_loss = min(averaged_val_losses)
        print(f"Averaged validation losses: {averaged_val_losses}")
        print(f"Minimum validation loss: {min_val_loss}")

        if len(averaged_val_losses) >= 5 and all(val > min_val_loss for val in averaged_val_losses[-5:]):
            print("Breaking loop as the last 5 values are greater than the minimum.")
            break

torch.save(model.state_dict(),'/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/DeepPhys.pt')

###############################################################################
#Test on train
train_denormalized_values_list_pred = []
train_denormalized_values_list_target = []
model = DeepPhys()
model.load_state_dict(torch.load('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/DeepPhys.pt', map_location='cpu'))
model.to('cuda')

predictions_train = []
targets_train = []
train_loss = 0.0

with torch.no_grad():
    model.eval()
    train_loss = 0.0
    predictions_train = []
    targets_train = []
    train_denormalized_values_list_pred = []
    train_denormalized_values_list_target = []
    for batch in tqdm(train_loader, desc='Training'):

        images, targets_norm, targets_original = batch
        images, targets_norm = images.to('cuda'), targets_norm.to('cuda')
        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)

        outputs = model(images)
        loss = torch.sqrt(criterion(outputs, targets_norm))
        train_loss += loss.sum().item()  # Somma le perdite del batch
        predictions_train.extend(outputs.cpu().numpy())
        targets_train.extend(targets_norm.cpu().numpy())

    train_loss /= len(train_loader.dataset)

    for batch_preds in predictions_train:
        if batch_preds.ndim > 0:
            for pred in batch_preds:
                denormalized_value = np.round(denormalize(pred, max_val, min_val), 2)
                train_denormalized_values_list_pred.append(denormalized_value)
        else:
            denormalized_value = np.round(denormalize(batch_preds, max_val, min_val), 2)
            train_denormalized_values_list_pred.append(denormalized_value)

    for batch_targets in targets_train:
        if batch_targets.ndim > 0:
            for target in batch_targets:
                denormalized_value = np.round(denormalize(target, max_val, min_val), 2)
                train_denormalized_values_list_target.append(denormalized_value)
        else:
            denormalized_value = np.round(denormalize(batch_targets, max_val, min_val), 2)
            train_denormalized_values_list_target.append(denormalized_value)

    with open("/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/EVM/output_train.txt", "w") as file:
        file.write(f"\nGround Truth: {train_denormalized_values_list_target}\n")
        file.write(f"Prediction: {train_denormalized_values_list_pred}\n")


mse = mean_squared_error(targets_train, predictions_train)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_train, predictions_train)
mean_targets = np.mean(targets_train)
targets_all_array_train = np.array(targets_train)
predictions_array_train = np.array(predictions_train)
mape = (np.mean(np.abs(targets_all_array_train - predictions_array_train) / targets_all_array_train)) * 100
residuals = np.array(targets_train) - np.array(predictions_train)
sde = np.std(residuals)
predictions_array_train = np.squeeze(predictions_array_train)
correlation_coef = np.corrcoef(targets_all_array_train, predictions_array_train)[1, 0]
r_squared = r2_score(targets_all_array_train, predictions_array_train)

print("\n--------------------------TRAINING METRICS--------------------------------\n")
print(f"RMSE: {rmse:.4f} MAE: {mae:.2f}, MAPE: {mape:.2f}, R: {correlation_coef:.2f}, r^2: {r_squared:.2f}, Standard Deviation of Error (SDe): {sde:.2f}")

plt.figure(figsize=(10, 8))
plt.scatter(targets_train, predictions_train, alpha=0.1, label='Predictions', color='red')
plt.scatter(targets_train, targets_train, alpha=0.1, label='Ground Truth', color='green')
plt.title('Scatter Plot: Ground Truth vs Prediction', fontsize=16)
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.legend(fontsize=12)
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/scatterplot_train_deep.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
errors = np.array(train_denormalized_values_list_target) - np.array(train_denormalized_values_list_pred)
plt.scatter(range(len(errors)), errors, alpha=0.1)
plt.title('Error Plot: Prediction Error for Each Example', fontsize=16)
plt.xlabel('Image', fontsize=14)
plt.ylabel('Prediction Error', fontsize=14)
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/errorplot_train_deep.png', dpi=300, bbox_inches='tight')
plt.close()


################################################################################

#Testing
model.eval()
predictions = []
targets_all = []
test_loss_list = []
denormalized_values_list_pred = []
denormalized_values_list_target = []

with torch.no_grad():
    model.eval()
    test_loss = 0.0
    predictions_dict = {}
    predictions_test = []
    targets_test = []
    denormalized_values_list_pred = []
    denormalized_values_list_target = []

    for batch in tqdm(test_loader, desc='Testing'):
        images, targets_norm, targets_original = batch
        images, targets_norm = images.to('cuda'), targets_norm.to('cuda')
        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)

        outputs = model(images)
        loss = torch.sqrt(criterion(outputs, targets_norm))
        test_loss += loss.sum().item()  # Somma le perdite del batch
        predictions_test.extend(outputs.cpu().numpy())
        targets_test.extend(targets_norm.cpu().numpy())

    test_loss /= len(test_loader.dataset)  # Calcola la perdita media

targets_all = targets_test
predictions = predictions_test

denormalized_values_list_pred = [np.round(denormalize(value, max_val, min_val), 2) for value in predictions]
denormalized_values_list_target = [np.round(denormalize(value, max_val, min_val), 2) for value in targets_all]
mse = mean_squared_error(targets_all, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_all, predictions)
mean_targets = np.mean(targets_all)
targets_all_array = np.array(targets_all)
predictions_array = np.array(predictions)
mape = (np.mean(np.abs(targets_all_array - predictions_array) / targets_all_array)) * 100
residuals = np.array(targets_all) - np.array(predictions)
sde = np.std(residuals)
predictions_array_test = np.squeeze(predictions_array)
correlation_coef = np.corrcoef(targets_all_array, predictions_array_test)[1, 0]
r_squared = r2_score(targets_all_array, predictions_array_test)


print("\n--------------------------TEST METRICS--------------------------------\n")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}, R: {correlation_coef:.2f}, r^2: {r_squared}, Standard Deviation of Error (SDe): {sde:.2f}")

for batch_preds in predictions_test:
    if batch_preds.ndim > 0:
        for pred in batch_preds:
            denormalized_value = np.round(denormalize(pred, max_val, min_val), 2)
            denormalized_values_list_pred.append(denormalized_value)
    else:
        denormalized_value = np.round(denormalize(batch_preds, max_val, min_val), 2)
        denormalized_values_list_pred.append(denormalized_value)

for batch_targets in targets_test:
    if batch_targets.ndim > 0:
        for target in batch_targets:
            denormalized_value = np.round(denormalize(target, max_val, min_val), 2)
            denormalized_values_list_target.append(denormalized_value)
    else:
        denormalized_value = np.round(denormalize(batch_targets, max_val, min_val), 2)
        denormalized_values_list_target.append(denormalized_value)

with open("/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/output_test.txt", "w") as file:
    file.write(f"\nGround Truth: {train_denormalized_values_list_target}\n")
    file.write(f"Prediction: {train_denormalized_values_list_pred}\n")

print("[INFO] - Best Hyperparameters:", best_params)

plt.figure(figsize=(12, 6))
plt.plot(denormalized_values_list_target, label='HR Original', marker='o', alpha=0.15)
plt.plot(denormalized_values_list_pred, label='Predictions', marker='x', alpha=0.15)
plt.title('True vs Predicted Values', fontsize=16)
plt.ylabel('Values', fontsize=14)
plt.legend(fontsize=12)
plt.xticks([])
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/true_vs_predicted_deep.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
plt.scatter(targets_all, predictions, alpha=0.1, label='Predictions', color='red')
plt.scatter(targets_all, targets_all, alpha=0.1, label='Ground Truth', color='green')
plt.title('Scatter Plot: Ground Truth vs Prediction', fontsize=16)
plt.xlabel('Ground Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.legend(fontsize=12)
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/scatterplot_deep.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
errors = np.array(train_denormalized_values_list_target) - np.array(train_denormalized_values_list_pred)
plt.scatter(range(len(errors)), errors, alpha=0.1)
plt.title('Error Plot: Prediction Error for Each Example', fontsize=16)
plt.xlabel('Image', fontsize=14)
plt.ylabel('Prediction Error', fontsize=14)
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/errorplot_deep.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(averaged_val_losses, label='RMSE', linewidth=2)
plt.legend(fontsize=12)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Root Mean Squared Error (RMSE) on Validation Set', fontsize=16)
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/rmse_val_plot_deep.png', dpi=300, bbox_inches='tight')
plt.close()


x_values = [100 * i for i in range(1, len(averaged_losses) + 1)]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x_values, averaged_losses, label='Train Loss')
ax.plot(x_values, averaged_val_losses, label='Validation Loss')

ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Train and validation losses', fontsize=16)
ax.legend()

plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/DEEP/mean_rmse_train_val_loss_deep.png')
plt.close()