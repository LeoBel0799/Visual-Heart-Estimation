
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

from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim,
                                               num_heads=n_heads,
                                               dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x, x, x)
        return attn_output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ViT(nn.Module):
    def __init__(self, ch=1, img_size=200, patch_size=None, emb_dim=None,
                 n_layers=None, out_dim=1, dropout=None, heads=None):
        super(ViT, self).__init__()

        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        self.patch_embedding = PatchEmbedding(in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, emb_dim, dropout=dropout))))
            self.layers.append(transformer_block)

        # Regression head
        self.head = nn.Linear(emb_dim, out_dim)

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        for i in range(self.n_layers):
            x = self.layers[i](x)

        return self.head(x[:, 0, :])




# class CustomDataset(Dataset):
#     def __init__(self, images_list, dataframe, transform=None):
#         self.images_list = images_list
#         self.df = dataframe
#         self.transform = transform
#
#         self.mean_hr_mapping = {i: self.df.loc[i, ' ECG HR'] for i in range(len(self.df))}
#         self.image_mapping = {i: img for i, img in enumerate(self.images_list)}
#         shared_indices = set(self.mean_hr_mapping.keys()) & set(self.image_mapping.keys())
#         shared_indices = sorted(shared_indices)
#         self.shared_data = [(self.image_mapping[idx], self.mean_hr_mapping[idx]) for idx in shared_indices]
#
#         self.max_hr = max(self.df[' ECG HR'])
#         self.min_hr = min(self.df[' ECG HR'])
#
#     def normalize_hr(self, hr):
#         return round((hr - self.min_hr) / (self.max_hr - self.min_hr), 3)
#
#     def denormalize_hr(self, norm_hr):
#         return round(norm_hr * (self.max_hr - self.min_hr) + self.min_hr, 3)
#
#     def __len__(self):
#         return len(self.shared_data)
#
#     def __getitem__(self, idx):
#         img, mean_hr = self.shared_data[idx]
#
#         if self.transform:
#             img = self.transform(img)
#
#         norm_mean_hr = self.normalize_hr(mean_hr)
#
#         return img, norm_mean_hr

class CustomDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

        self.max_hr = max(hr for _, hr in self.images_list)
        self.min_hr = min(hr for _, hr in self.images_list)

    def normalize_hr(self, hr):
        return round((hr - self.min_hr) / (self.max_hr - self.min_hr), 3)

    def denormalize_hr(self, norm_hr):
        return round(norm_hr * (self.max_hr - self.min_hr) + self.min_hr, 3)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img, mean_hr = self.images_list[idx]

        if self.transform:
            img = self.transform(img)

        norm_mean_hr = self.normalize_hr(mean_hr)
        denorm_mean_hr = self.denormalize_hr(norm_mean_hr)
        return img, norm_mean_hr, denorm_mean_hr

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

        if len(face_region.shape) == 3:
            feature_image = cv2.resize(face_region, (200, 200))
            feature_image = np.transpose(feature_image, (2, 0, 1))
        else:
            feature_image = cv2.resize(face_region, (200, 200))
            #feature_image_res = np.expand_dims(feature_image, axis=0)

        return feature_image
    else:
        print("Region outside image limits.")
        return None

def process_video(video_path, video_csv_path, face_detector, landmark_predictor):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None

        rois_list = []

        # Load ECG data
        ecg_data = pd.read_csv(video_csv_path, index_col='milliseconds')
        ecg_timestamps = ecg_data.index

        # Calculate the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        max_time_to_analyze_seconds = 58  # Adjust the desired time duration in seconds
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

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray_frame = cv2.resize(gray_frame, (300, 300))
            faces = face_detector(gray_frame, 1)

            if not faces:
                frame_count += 1
                continue

            face = faces[0]
            landmarks = landmark_predictor(gray_frame, face.rect)

            face_region = extract_face_region(gray_frame, landmarks)
            if face_region is not None:
                video_frame_timestamp = frame_count * (1000 / frame_rate)
                closest_timestamp = min(ecg_timestamps, key=lambda x: abs(x - video_frame_timestamp))

                ecg_value = ecg_data.at[closest_timestamp, " ECG HR"]

                rois_list.append((face_region, ecg_value))

            progress_bar.update(1)
            frame_count += 1

        progress_bar.close()

        print(f"Video analyzed for {video_path}")
        current_dataset = CustomDataset(rois_list)  # Assuming CustomDataset is defined somewhere in your code
        return current_dataset
    except Exception as e:
        print(f"Error processing video: {video_path}. Error details: {str(e)}")
        return None  # Returning None to indicate an error
    finally:
        cap.release()




def process_and_create_dataset (main_directory, video_to_process):
    main_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    face_detector = dlib.cnn_face_detection_model_v1("/home/ubuntu/data/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat")
    landmark_predictor = dlib.shape_predictor(
        "/home/ubuntu/data/ecg-fitness_raw-v1.0/shape_predictor_68_face_landmarks_GTX.dat")
    processed_videos = 0
    all_datasets = []
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

                if len(fin_csv_files) == 1:
                    fin_csv_file = fin_csv_files[0]
                    video_csv_path = os.path.join(sub_dir_path, fin_csv_file)
                else:
                    print(f"Error: No or multiple 'fin' CSV files found in {sub_dir_path}")
                    continue

                current_dataset = process_video(video_path, video_csv_path, face_detector, landmark_predictor)

                if current_dataset is not None:
                    current_dataset_length = len(current_dataset)
                    all_datasets.append(current_dataset)
                    print(f"All datasets len: {len(all_datasets)}")
                    print(f"CustomDataset created for {video_path} with {current_dataset_length} rows")
                    processed_videos += 1
                else:
                    print(f"Skipping video {video_path} due to processing error.")

    combined_dataset = ConcatDataset(all_datasets)

    print("Global custom dataset created with length: ", len(combined_dataset))
    return combined_dataset

main_directory = "/home/ubuntu/data/ecg-fitness_raw-v1.0"
#video_to_process = 70
dataset_path = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/combined_dataset.pth"
#final_dataset = process_and_create_dataset(main_directory,video_to_process)
#torch.save(final_dataset, dataset_path)
#print("Global custom dataset saved!")
loaded_dataset = torch.load(dataset_path)
print("Global custom dataset loaded...!")


train_size = int(0.8 * len(loaded_dataset))
val_size = int(0.1 * len(loaded_dataset))
test_size = len(loaded_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(loaded_dataset, [train_size, val_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

total_samples_in_train_loader = len(train_loader.dataset)
print(f"Total number of samples in train_loader: {total_samples_in_train_loader}")

total_samples_in_val_loader = len(val_loader.dataset)
print(f"Total number of samples in val_loader: {total_samples_in_val_loader}")

total_samples_in_test_loader = len(test_loader.dataset)
print(f"Total number of samples in test_loader: {total_samples_in_test_loader}")



#-----CODICE PER VISUALIZZARE IMMAGINI E LABEL (COLAB)-------------
# # Assume che il tuo DataLoader sia chiamato train_loader
# for batch_idx, (images, norm_targets, denorm_targets) in enumerate(train_loader):
#     if batch_idx >= 4:
#         break  # Esce dopo i primi 4 batch
#
#     print(f"Train Batch {batch_idx + 1} - Images shape: {images.shape}, Targets shape: {norm_targets.shape}")
#
#     # Estrai le immagini e i target dal batch
#     batch_images = images.numpy()
#     batch_norm_targets = norm_targets.numpy()
#
#     # Visualizza solo 3 immagini per batch
#     num_images_to_display = min(3, batch_images.shape[0])
#     for i in range(num_images_to_display):
#         plt.subplot(1, 3, i + 1)  # Organizza in una griglia 1x3
#         plt.imshow(np.squeeze(batch_images[i]))  # Trasponi per adattarlo a imshow
#         plt.title(f'Target: {batch_norm_targets[i]}')
#
#     plt.show()
#
#
# # Assume che il tuo DataLoader sia chiamato train_loader
# for batch_idx, (images, norm_targets, denorm_targets) in enumerate(train_loader):
#     if batch_idx >= 4:
#         break  # Esce dopo i primi 4 batch
#
#     print(f"Train Batch {batch_idx + 1} - Images shape: {images.shape}, Norm Targets shape: {norm_targets.shape}, Denorm Targets shape: {denorm_targets.shape}")
#
#     # Estrai le immagini e i target dal batch
#     batch_images = images.numpy()
#     batch_denorm_targets = denorm_targets.numpy()
#
#     # Visualizza solo 3 immagini per batch
#     num_images_to_display = min(3, batch_images.shape[0])
#     for i in range(num_images_to_display):
#         plt.subplot(1, 3, i + 1)  # Organizza in una griglia 1x3
#         plt.imshow(np.squeeze(batch_images[i]))  # Trasponi per adattarlo a imshow
#         denormalized_target = batch_denorm_targets[i]
#         plt.title(f'Denormalized Target: {denormalized_target}')
#
#     plt.show()
#---------------------------------------------------------------------------------------------





N_EPOCHS = 50
LR = 0.001
VAL_EVERY = 3
PATIENCE = 5
WD = 0.01

# Creazione del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)

criterion = nn.MSELoss()


def objective(trial, train_loader, test_loader):
    patch_size = trial.suggest_categorical('patch_size', [1, 2, 4])

    # Genera 'emb_dim' come multiplo di 'heads'
    heads = trial.suggest_int('heads', 1, 4)
    emb_dim = heads * trial.suggest_int('emb_dim', 1, 8)

    n_layers = trial.suggest_int('n_layers', 4, 12)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    print(f"Trying parameters: patch_size={patch_size}, heads={heads}, emb_dim={emb_dim}, n_layers={n_layers}, dropout={dropout}")


    model = ViT(patch_size=patch_size, emb_dim=emb_dim, n_layers=n_layers, dropout=dropout, heads=heads)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()


    N_EPOCHS = 8
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} in training", leave=True):
            images, targets, hr = batch

            for image, target in zip(images, targets):
                # Trasferisci l'immagine e il target sulla GPU
                image = image.float().cuda()
                target = target.float().cuda()
                image = image.unsqueeze(0)
                target = target.unsqueeze(0)
                image = image.unsqueeze(0)
                target = target.unsqueeze(0)
                outputs = model(image)
                loss = criterion(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Train RMSE: {math.sqrt(train_loss):.4f}")

        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for test_batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} in validation", leave=True):
                val_images, val_targets, val_hr = test_batch

                for val_image, val_target in zip(val_images, val_targets):
                    val_image = val_image.float().cuda()
                    val_target = val_target.float().cuda()
                    val_image = val_image.unsqueeze(0)
                    val_target = val_target.unsqueeze(0)
                    val_image = val_image.unsqueeze(0)
                    val_target = val_target.unsqueeze(0)
                    val_outputs = model(val_image)
                    test_loss += criterion(val_outputs, val_target).item()

        test_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Test RMSE: {math.sqrt(test_loss):.4f}")

    return math.sqrt(test_loss)

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_loader, test_loader), n_trials=5)

best_params = study.best_params
print("Best Hyperparameters:", best_params)

best_patch_size = best_params['patch_size']
best_emb_dim = best_params['emb_dim']
best_n_layers = best_params['n_layers']
best_dropout = best_params['dropout']
best_heads = best_params['heads']

best_model = ViT(patch_size=best_patch_size, emb_dim=best_emb_dim, n_layers=best_n_layers, dropout=best_dropout, heads = best_heads)
print(best_model)

best_model.to(device)

optimizer = AdamW(best_model.parameters(), lr=LR, weight_decay=WD)

best_val_rmse = float('inf')
no_improvement_count = 0

train_loss_list = []
val_loss_list = []
rmse_list = []
me_rate_list = []
pearson_correlation_list = []

for epoch in range(N_EPOCHS):
    best_model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} in training", leave=True):
        images, targets, hr = batch

        for image, target in zip(images, targets):
            image = image.float().cuda()
            target = target.float().cuda()
            image = image.unsqueeze(0)
            target = target.unsqueeze(0)
            target = target.unsqueeze(0)
            # Aggiungi una dimensione di batch per gestire la singola immagine
            image = image.unsqueeze(0)
            #print(image.shape)
            #print(target)
            outputs = best_model(image)

            #print(outputs)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{N_EPOCHS}, Train RMSE: {math.sqrt(train_loss):.4f}")

    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    if (epoch + 1) % VAL_EVERY == 0:
        best_model.eval()
        val_loss = 0.0
        predictions = []
        targets_all = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} in validation", leave=True):
                val_images, val_targets, val_hr = batch
                batch_predictions = []

                for val_image, val_target in zip(val_images, val_targets):
                    val_image = val_image.float().cuda()
                    val_target = val_target.float().cuda()
                    val_image = val_image.unsqueeze(0)
                    val_target = val_target.unsqueeze(0)
                    val_target = val_target.unsqueeze(0)
                    val_image = val_image.unsqueeze(0)

                    val_outputs = best_model(val_image)
                    #print(f"Attuale: ", val_target)
                    #print(f"Previsto: ", val_outputs)
                    val_loss += criterion(val_outputs, val_target).item()
                    batch_predictions.append(val_outputs.cpu().numpy())
                    targets_all.append(val_target.cpu().numpy())

                predictions.extend(np.concatenate(batch_predictions, axis=0))

        val_loss /= len(val_loader.dataset)
        val_rmse = math.sqrt(val_loss)

        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Validation RMSE: {val_rmse:.4f}")

        # Early stopping check
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs. Early stopping.")
                break  # Spostato il break qui

avg_val_loss = val_loss / len(val_loader)
val_loss_list.append(avg_val_loss)

# RMSE
rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets_all))**2))
rmse_list.append(rmse)

# Mean Error (Me)
mean_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)))

# Standard Deviation Error (SDe)
std_dev_error = np.sqrt(np.mean((np.array(predictions) - np.array(targets_all) - mean_error)**2))

# Mean Absolute Percentage Error (MeRate)
absolute_target_values = np.abs(np.array(targets_all))
me_rate_list.append(np.mean(np.divide(np.abs(np.array(predictions) - np.array(targets_all)),
                                      np.where(absolute_target_values == 0, 1, absolute_target_values))))



print(f'Epoch [{epoch + 1}/{N_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, MeRate: {me_rate_list[-1]:.4f}')

best_model.eval()
test_loss = 0.0
test_predictions = []
test_targets_all = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc=f"Testing", leave=True):
        test_images, test_targets, test_hr = batch
        batch_predictions = []

        for test_image, test_target in zip(test_images, test_targets):
            test_image = test_image.float().cuda()
            test_target = test_target.float().cuda()
            test_image = test_image.unsqueeze(0)
            test_target = test_target.unsqueeze(0)
            test_target = test_target.unsqueeze(0)
            test_image = test_image.unsqueeze(0)
            test_outputs = best_model(test_image)
            #print(f"Attuale: ", test_target)
            #print(f"Previsto: ", test_outputs)
            # Compute and accumulate the test loss
            test_loss += criterion(test_outputs, test_target).item()

            # Convert the predictions and targets to numpy arrays
            batch_predictions.append(test_outputs.cpu().numpy())
            test_targets_all.append(test_target.cpu().numpy())

        # Extend the test_predictions list with the batch predictions
        test_predictions.extend(np.concatenate(batch_predictions, axis=0))

# Calculate average test loss
test_loss /= len(test_loader.dataset)
test_rmse = math.sqrt(test_loss)

print(f"Test RMSE: {test_rmse:.4f}")

for true_value, predicted_value in zip(test_targets_all, test_predictions):
    print(f"True Value: {true_value}, Predicted Value: {predicted_value}")


model_save_path = '/content/drive/MyDrive/Thesis <BELLIZZI>/ecg-fitness_raw-v1.0/trained_VIT.pth'
torch.save(best_model, model_save_path)
#loaded_model = torch.load(model_save_path)
#loaded_model.to(device)


predictions = np.concatenate(predictions, axis=0)
targets_all = np.concatenate(targets_all, axis=0)

# Plotting true values and predicted values
plt.figure(figsize=(10, 6))
plt.plot(targets_all, label='True Values', marker='o')
plt.plot(predictions, label='Predicted Values', marker='x')
plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/true_vs_predicted.png')
plt.close()

plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/train_val_loss_vit.png')
plt.close()

plt.plot(rmse_list, label='RMSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) on Validation Set')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/rmse_plot_vit.png')
plt.close()

plt.plot(me_rate_list, label='Mean Absolute Percentage Error')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Percentage Error')
plt.title('Mean Absolute Percentage Error on Validation Set')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/mape_plot_vit.png')
plt.close()

plt.plot(pearson_correlation_list, label="Pearson's Correlation Coefficient")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Pearson's Correlation Coefficient")
plt.title("Pearson's Correlation Coefficient on Validation Set")
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/pearson_corr_plot_vit.png')
plt.close()



# def objective(trial, train_loader, test_loader):
#     # Definisci gli hyperparameters suggeriti da Optuna
#     patch_size = trial.suggest_int('patch_size', 4, 8)
#     emb_dim = trial.suggest_int('emb_dim', 16, 128)
#     n_layers = trial.suggest_int('n_layers', 4, 12)
#     dropout = trial.suggest_float('dropout', 0.0, 0.5)
#     heads = trial.suggest_int('heads', 1, 3)
#     emb_dim = emb_dim + (heads - emb_dim % heads) % heads
#
#     # Crea il modello con gli hyperparameters suggeriti
#     model = ViT(patch_size=patch_size, emb_dim=emb_dim, n_layers=n_layers, dropout=dropout, heads=heads)
#     model.to(device)
#
#     optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
#     criterion = nn.MSELoss()
#
#     # Addestra il modello per un numero fisso di epoche
#     N_EPOCHS = 8
#     for epoch in range(N_EPOCHS):
#         model.train()
#         train_loss = 0.0
#
#         for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} in training", leave=True):
#             images, targets = batch
#
#             for image, target in zip(images, targets):
#                 # Trasferisci l'immagine e il target sulla GPU
#                 image = image.float().cuda()
#                 target = target.float().cuda()
#                 image = image.unsqueeze(0)
#                 target = target.unsqueeze(0)
#                 image = image.unsqueeze(0)
#
#                 outputs = model(image)
#                 loss = criterion(outputs, target)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             train_loss += loss.item()
#
#         train_loss /= len(train_loader.dataset)
#         print(f"Epoch {epoch + 1}/{N_EPOCHS}, Train RMSE: {math.sqrt(train_loss):.4f}")
#
#         # Test the model after each epoch
#         model.eval()
#         test_loss = 0.0
#
#         with torch.no_grad():
#             for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS} in testing", leave=True):
#                 images, targets = batch
#
#                 for image, target in zip(images, targets):
#                     # Trasferisci l'immagine e il target sulla GPU
#                     image = image.float().cuda()
#                     target = target.float().cuda()
#                     image = image.unsqueeze(0)
#                     target = target.unsqueeze(0)
#                     image = image.unsqueeze(0)
#                     # Forward pass
#                     outputs = model(image)
#                     loss = criterion(outputs, targets)
#                 test_loss += loss.item()
#
#                 # Print 5 real and predicted values during testing
#                 # Print 5 real and predicted values during testing
#                 if len(test_loader) == 5:
#                     print("Testing - Epoch {}, Real vs Predicted:".format(epoch + 1))
#                     for i in range(5):
#                         real_value = CustomDataset.denormalize_hr(target[i].item())
#                         predicted_value = CustomDataset.denormalize_hr(outputs[i].item())
#
#                         print("Real: {:.4f}\tPredicted: {:.4f}".format(real_value, predicted_value))
#
#         test_loss /= len(test_loader.dataset)
#         print(f"Epoch {epoch + 1}/{N_EPOCHS}, Test RMSE: {math.sqrt(test_loss):.4f}")
#
#     return math.sqrt(test_loss)  # Restituisci il RMSE come metrica di valutazione




# def process_video(video_path, video_csv_path, face_detector, landmark_predictor):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             print(f"Error opening video file: {video_path}")
#             return
#
#         rois_list = []
#
#         # Load ECG data
#         ecg_data = pd.read_csv(video_csv_path)
#         ecg_timestamps = ecg_data["milliseconds"]
#
#         ret, frame = cap.read()
#
#         # Calculate the total number of frames in the video
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_rate = cap.get(cv2.CAP_PROP_FPS)
#
#         max_time_to_analyze_seconds = 58  # Adjust the desired time duration in seconds
#         max_frames_to_analyze = int(max_time_to_analyze_seconds * frame_rate)
#         df = pd.read_csv(video_csv_path)
#         df = df[df['milliseconds'] <= max_time_to_analyze_seconds * 1000]
#         progress_bar = tqdm(total=min(total_frames, max_frames_to_analyze), position=0, leave=True,
#                             desc=f'Processing Frames for {video_path}')
#
#         frame_count = 0
#
#         while True:
#             ret, frame = cap.read()
#
#             if not ret or frame_count >= max_frames_to_analyze:
#                 break
#
#             frame = cv2.resize(frame, (300, 300))
#             faces = face_detector(frame, 1)
#
#             if not faces:
#                 frame_count += 1
#                 continue
#
#                 # Use only the first detected face
#             face = faces[0]
#             landmarks = landmark_predictor(frame, face.rect)
#
#             face_region = extract_face_region(frame, landmarks)
#             if face_region is not None:
#                 # Find the closest timestamp in ECG data
#                 video_frame_timestamp = frame_count * (1000 / frame_rate)
#                 closest_timestamp = min(ecg_timestamps, key=lambda x: abs(x - video_frame_timestamp))
#                 ecg_value = ecg_data.loc[ecg_data["milliseconds"] == closest_timestamp, " ECG HR"].values[0]
#
#                 # Add the face region and associated ECG value to the list
#                 rois_list.append((face_region, ecg_value))
#
#             progress_bar.update(1)
#             frame_count += 1
#
#         progress_bar.close()
#
#         print(f"Video analyzed for {video_path}")
#         current_dataset = CustomDataset(rois_list, df)  # Assuming CustomDataset is defined somewhere in your code
#         return current_dataset
#     except Exception as e:
#         print(f"Error processing video: {video_path}. Error details: {str(e)}")
#         return None  # Returning None to indicate an error
#     finally:
#         cap.release()



# model = model.cuda()
# criterion = nn.MSELoss()
#
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 30
#
# train_loss_list = []
# val_loss_list = []
# rmse_list = []
# me_rate_list = []
# pearson_correlation_list = []
#
# for epoch in range(num_epochs):
#     # Addestramento
#     model.train()
#     train_loss = 0.0
#     for images, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#         images = images.float().cuda()
#         targets = targets.float().cuda()
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, targets)
#
#         # Backward pass e ottimizzazione
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#     # Calcola la perdita media per epoca di addestramento
#     avg_train_loss = train_loss / len(train_loader)
#     train_loss_list.append(avg_train_loss)
#
#     # Validazione
#     model.eval()
#     val_loss = 0.0
#     predictions = []
#     targets_all = []
#     with torch.no_grad():
#         for images, targets in tqdm(val_loader, desc='Validation'):
#             # Converti i dati in float32
#             images = images.float().cuda()
#             targets = targets.float().cuda()
#
#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()
#
#             predictions.extend(outputs.cpu().numpy())
#             targets_all.extend(targets.cpu().numpy())
#
#     #plot_predictions(targets_all, predictions, f'Validation - Predicted vs Ground Truth')
#
#     # Calcola la perdita media per epoca di validazione
#     avg_val_loss = val_loss / len(val_loader)
#     val_loss_list.append(avg_val_loss)
#
#     #  RMSE
#     rmse = np.sqrt(((np.array(predictions) - np.array(targets_all))**2).mean())
#     rmse_list.append(rmse)
#
#     # Media dell'errore di misura (Me)
#     mean_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)))
#
#     # Deviazione standard dell'errore di misura (SDe)
#     std_dev_error = np.sqrt(np.mean((np.array(predictions) - np.array(targets_all) - mean_error)**2))
#
#     # Calcola il Mean Absolute Percentage Error (MeRate)
#     mean_absolute_percentage_error = np.mean(np.abs(np.array(predictions) - np.array(targets_all)) / np.abs(np.array(targets_all)))
#     me_rate_list.append(mean_absolute_percentage_error)
#
#     # Calcola Pearson's Correlation Coefficient (?)
#     mean_ground_truth = np.mean(np.array(targets_all))
#     mean_predicted_hr = np.mean(np.array(predictions))
#     numerator = np.sum((np.array(targets_all) - mean_ground_truth) * (np.array(predictions) - mean_predicted_hr))
#     denominator_ground_truth = np.sum((np.array(targets_all) - mean_ground_truth)**2)
#     denominator_predicted_hr = np.sum((np.array(predictions) - mean_predicted_hr)**2)
#     pearson_correlation = numerator / np.sqrt(denominator_ground_truth * denominator_predicted_hr)
#     pearson_correlation_list.append(pearson_correlation)
#
#     # Stampa le perdite e le metriche per ogni epoca
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, MeRate: {mean_absolute_percentage_error:.4f}, Pearson Correlation: {pearson_correlation:.4f}')
#
#
# # Test
# model.eval()
# test_loss = 0.0
# test_predictions = []
# test_targets_all = []
#
# with torch.no_grad():
#     for images, targets in tqdm(test_loader, desc='Testing'):
#         # Converti le immagini in float
#         images = images.float().cuda()
#         targets = targets.float().cuda()
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, targets)
#         test_loss += loss.item()
#
#         # Salva le previsioni e i target per ulteriori analisi
#         test_predictions.extend(outputs.cpu().numpy())
#         test_targets_all.extend(targets.cpu().numpy())
#
# # Calcola la perdita media per l'epoca di test
# avg_test_loss = test_loss / len(test_loader)
# print(f'Test Loss: {avg_test_loss:.4f}')
#
# #  RMSE
# rmse_test = np.sqrt(((np.array(test_predictions) - np.array(test_targets_all))**2).mean())
# print(f'Test RMSE: {rmse_test:.4f}')
#
# plt.plot(test_targets_all, label='Ground Truth')
# plt.plot(test_predictions, label='Predicted')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.legend()
# plt.title('Test Set: Predicted vs Ground Truth Over Time')
# plt.show()
#
#
# torch.save(model, 'home/ubuntu/ecg-fitness_raw-v1.0/CNNRegressorMIXED.pth')
