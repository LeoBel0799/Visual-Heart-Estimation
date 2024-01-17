
from facenet_pytorch import *
import neurokit2 as nk
from torch.cuda.amp import autocast, GradScaler
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
from einops.layers.torch import Rearrange
import warnings

warnings.filterwarnings("ignore")

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=None):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=None, n_blocks=None, hidden_d=None, n_heads=None, out_d=1):
        super(MyViT, self).__init__()

        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_d = hidden_d
        self.n_heads = n_heads


        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, hidden_d)
        self.class_token = nn.Parameter(torch.randn(1, hidden_d))
        self.register_buffer("positional_embeddings", get_positional_embeddings(n_patches**2 + 1, hidden_d), persistent=False)
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.ffnn = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, out_d)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]  # Get only the classification token

        # Apply the feed-forward neural network for regression
        output = self.ffnn(out)

        return output


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            angle = torch.tensor(i / (10000 ** (j / d)), dtype=torch.float32)
            result[i][j] = torch.sin(angle) if j % 2 == 0 else torch.cos(angle)
    return result


class CustomDataset(Dataset):
    def __init__(self, images_list, rows, transform=None):
        self.images_list = images_list
        self.df = rows
        self.transform = transform

        self.num_images = len(self.images_list)
        self.ecg_hr_values = self.df[' ECG HR'].tolist()


    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img = self.images_list[idx]
        ecg_hr = self.ecg_hr_values[idx]

        return img, ecg_hr


class CustomDatasetNormalized(Dataset):
    def __init__(self, img_and_label):
        self.img_and_label = img_and_label

    def __len__(self):
        return len(self.img_and_label)

    def __getitem__(self, idx):
        img, hr_norm, hr_original = self.img_and_label[idx]
        try:
            hr_norm = float(hr_norm)
            hr_original = float(hr_original)
        except ValueError:
            raise ValueError(f"Errore: hr_norm o hr_original are not numbers in index: {idx}")


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

        if len(face_region.shape) == 3:
            feature_image = cv2.resize(face_region, (200, 200))
            feature_image = np.transpose(feature_image, (2, 0, 1))
        else:
            feature_image = cv2.resize(face_region, (200, 200))
            # feature_image_res = np.expand_dims(feature_image, axis=0)

        return feature_image
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


Pl = 4
Fps = 30
Fl = 0.75
Fh = 4


def process_video(video_path, video_csv_path, face_detector, landmark_predictor, tracker, Pl, Fps, Fl, Fh):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    rois_list = []
    ecg_data = pd.read_csv(video_csv_path, index_col='milliseconds')
    ecg_timestamps = ecg_data.index

    # Calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Limit the number of frames to analyze
    max_time_to_analyze_seconds = 30  # Adjust the desired time duration in seconds
    sampling_interval_ms = 10

    max_frames_to_analyze = int(max_time_to_analyze_seconds * frame_rate)
    df = pd.read_csv(video_csv_path)
    # Filtra il DataFrame per ottenere solo le righe entro i primi tot secondi
    df = df[df['milliseconds'] <= max_time_to_analyze_seconds * 1000]
    selected_rows = df[df['milliseconds'] % sampling_interval_ms == 0]
    progress_bar = tqdm(total=min(total_frames, max_frames_to_analyze), position=0, leave=True,
                        desc=f'Processing Frames for {video_path}')

    frame_count = 0
    video_images = []

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= max_frames_to_analyze:
            break

        if frame_count % 10 == 0:
            faces = face_detector(frame, 1)

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


        progress_bar.update(1)
        frame_count += 1

    progress_bar.close()
    print(f"Video analyzed for {video_path}")
    features_img = extract_features(rois_list, Pl, Fps, Fl, Fh)
    video_images.extend(features_img)
    print(f"Features extracted for {video_path}")
    current_dataset = CustomDataset(video_images,selected_rows)

    return current_dataset


def process_and_create_dataset(main_directory, video_to_process):
    main_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    face_detector = dlib.cnn_face_detection_model_v1(
        "/home/ubuntu/data/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat")
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
                for csv_file in fin_csv_files:
                    file_path = os.path.join(sub_dir_path, csv_file)
                    df = pd.read_csv(file_path)
                    df = df[df[' ECG HR'] >= 0]
                    df.to_csv(file_path, index=False)

                if len(fin_csv_files) == 1:
                    fin_csv_file = fin_csv_files[0]
                    video_csv_path = os.path.join(sub_dir_path, fin_csv_file)
                else:
                    print(f"Error: No or multiple 'fin' CSV files found in {sub_dir_path}")
                    continue

                tracker = cv2.TrackerGOTURN_create()

                current_dataset = process_video(video_path, video_csv_path, face_detector, landmark_predictor, tracker,
                                                Pl, Fps, Fl, Fh)

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


def normalize(custom_dataset, normalized_dataset):
    min_mean_hr = float('inf')
    max_mean_hr = float('-inf')

    for _, mean_hr in custom_dataset:
        min_mean_hr = min(min_mean_hr, mean_hr)
        max_mean_hr = max(max_mean_hr, mean_hr)

    for i in range(len(custom_dataset)):
        img, mean_hr = custom_dataset[i]
        norm_mean_hr = (mean_hr - min_mean_hr) / (max_mean_hr - min_mean_hr)
        normalized_dataset.append([img, norm_mean_hr, mean_hr])
    return CustomDatasetNormalized(normalized_dataset), min_mean_hr, max_mean_hr


def denormalize(y, max_v, min_v):
    final_value = y * (max_v - min_v) + min_v
    return final_value


main_directory = "/home/ubuntu/data/ecg-fitness_raw-v1.0"
video_to_process = 4
dataset_path = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Dataset/ViT.pth"
final_dataset = process_and_create_dataset(main_directory, video_to_process)
norm_dataset = []
normalized_dataset, min_for_denorm, max_for_denorm = normalize(final_dataset, norm_dataset)


torch.save(normalized_dataset, dataset_path)
print("Global custom dataset saved!")
loaded_dataset = torch.load(dataset_path)
print("Global custom dataset loaded...!")

train_size = int(0.8 * len(loaded_dataset))
val_size = int(0.1 * len(loaded_dataset))
test_size = len(loaded_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(loaded_dataset, [train_size, val_size, test_size])

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



total_samples_in_train_loader = len(train_loader.dataset)
print(f"Total number of samples in train_loader: {total_samples_in_train_loader}")

total_samples_in_val_loader = len(val_loader.dataset)
print(f"Total number of samples in val_loader: {total_samples_in_val_loader}")

total_samples_in_test_loader = len(test_loader.dataset)
print(f"Total number of samples in test_loader: {total_samples_in_test_loader}")


N_EPOCHS = 50
LR = 0.001
PATIENCE = 5
WD = 0.01

# Creazione del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device,f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",)

criterion = nn.MSELoss()



def objective(trial, train, val):
    best_val_rmse = float('inf')
    val_loss_list = []
    n_patches = trial.suggest_categorical('n_patches', [1, 2, 4, 8])
    n_heads = trial.suggest_categorical('n_heads', [1,2,4])
    hidden_d = trial.suggest_categorical('hidden_d', [4,8,12])
    n_blocks = trial.suggest_int('n_block', 2,8)

    print(f"Trying parameters: patch_size={n_patches}, heads={n_heads}, hidden_d={hidden_d}, n_blocks={n_blocks}")


    model = MyViT((3,40,40), n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d,n_heads=n_heads)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()

    N_EPOCHS = 5
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        scaler = GradScaler()

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{N_EPOCHS}'):
            images, targets_norm, targets_original = batch
            images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(
                device)

            images = images.to(torch.float32)
            targets_norm = targets_norm.to(torch.float32)
            targets_original = targets_original.to(torch.float32)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets_norm)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")


        model.eval()
        predictions = []
        targets_all = []
        with torch.no_grad():
            val_loss = 0.0

            for batch in tqdm(val_loader, desc='Validation'):
                images, targets_norm, targets_original = batch
                images, targets_norm, targets_original = images.to(device), targets_norm.to(
                    device), targets_original.to(device)

                images = images.to(torch.float32)
                targets_norm = targets_norm.to(torch.float32)
                targets_original = targets_original.to(torch.float32)

                outputs = model(images)
                loss = criterion(outputs, targets_norm)

                val_loss += loss.item() / len(val_loader)
                predictions.extend(outputs.cpu().numpy())
                targets_all.extend(targets_norm.cpu().numpy())


        mse = mean_squared_error(targets_all, predictions)
        rmse_validation = np.sqrt(mse)
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS}, Validation RMSE: {rmse_validation:.4f},  Avg validation Loss: {avg_val_loss}")
        # Early stopping check
        if rmse_validation < best_val_rmse:
            best_val_rmse = rmse_validation
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs. Early stopping.")
                break

    return rmse_validation

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=3)

best_params = study.best_params
print("Best Hyperparameters:", best_params)

best_patch_size = best_params['n_patches']
best_heads = best_params['n_heads']
best_hidden = best_params['hidden_d']
best_nblock = best_params['n_block']

best_model = MyViT((3,40,40), n_patches=best_patch_size, n_blocks=best_nblock, hidden_d=best_hidden,  n_heads=best_heads)
print(best_model)

best_model.to(device)

optimizer = AdamW(best_model.parameters(), lr=LR, weight_decay=WD)

best_val_rmse = float('inf')
no_improvement_count = 0

train_loss_list = []
val_loss_list = []
rmse_list_val = []
best_val_rmse = float('inf')
no_improvement_count = 0
best_model.to(device)
normalized_value_list_validation = []
denormalized_values_list_validation = []

for epoch in range(N_EPOCHS):
    best_model.train()
    train_loss = 0.0
    scaler = GradScaler()

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{N_EPOCHS}'):
        images, targets_norm, targets_original = batch
        images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(device)

        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)
        targets_original = targets_original.to(torch.float32)

        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)
        targets_original = targets_original.to(torch.float32)

        with autocast():
            outputs = best_model(images)
            loss = criterion(outputs, targets_norm)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    best_model.eval()
    predictions = []
    targets_all = []
    with torch.no_grad():
        val_loss = 0.0

        for batch in tqdm(val_loader, desc='Validation'):
            images, targets_norm, targets_original = batch
            images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(
                device)

            images = images.to(torch.float32)
            targets_norm = targets_norm.to(torch.float32)
            targets_original = targets_original.to(torch.float32)

            outputs = best_model(images)
            loss = criterion(outputs, targets_norm)

            val_loss += loss.item() / len(val_loader)
            predictions.extend(outputs.cpu().numpy())
            targets_all.extend(targets_norm.cpu().numpy())

    mse = mean_squared_error(targets_all, predictions)
    rmse_validation = np.sqrt(mse)
    rmse_list_val.append(rmse_validation)
    avg_val_loss = val_loss / len(val_loader)
    val_loss_list.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{N_EPOCHS}, Validation RMSE: {rmse_validation:.4f},  Avg validation Loss: {avg_val_loss}")
    # Early stopping check
    if rmse_validation < best_val_rmse:
        best_val_rmse = rmse_validation
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs. Early stopping.")
            break



# Testing
best_model.eval()
predictions = []
targets_all = []
predictions_test = []
targets_all_test = []
test_loss_list = []

normalized_value_list = []
denormalized_values_list_pred = []
denormalized_values_list_target = []


with torch.no_grad():
    test_loss = 0.0
    for batch in tqdm(test_loader, desc='Testing'):
        images, targets_norm, targets_original = batch
        images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(device)

        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)
        targets_original = targets_original.to(torch.float32)

        outputs = best_model(images)
        loss = criterion(outputs, targets_norm)
        test_loss += loss.item() / len(test_loader)

        predictions.extend(outputs.cpu().numpy())
        targets_all.extend(targets_norm.cpu().numpy())


    for value in predictions:
        denormalized_value = np.round(denormalize(value[0], max_for_denorm, min_for_denorm), 2)
        denormalized_values_list_pred.append(denormalized_value)

    for value in targets_all:
        denormalized_value = np.round(denormalize(value, max_for_denorm, min_for_denorm), 2)
        denormalized_values_list_target.append(denormalized_value)


mse = mean_squared_error(targets_all, predictions)
rmse = np.sqrt(mse)

print(f"Test RMSE: {rmse:.4f}, Test Loss: {test_loss:.2f}")

torch.save(best_model, '/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Model/VIT.pth')
print(f"Ground Truth:", denormalized_values_list_target)
print("Prediction:", denormalized_values_list_pred)

plt.plot(denormalized_values_list_target, label='HR Original', marker='o')
plt.plot(denormalized_values_list_pred, label='Predictions', marker='x')
plt.title('True vs Predicted Values')
plt.ylabel('Predictions')
plt.legend()
plt.xticks([])
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Plot/true_vs_predicted_VIT.png')
plt.close()

plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Plot/train_val_loss_VIT.png')
plt.close()


plt.scatter(targets_all, predictions, alpha=0.5, label='Predictions', color='red')
plt.scatter(targets_all, targets_all, alpha=0.5, label='Ground Truth', color='green')
plt.title('Scatter Plot: Ground Truth vs Prediction')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Plot/scatterplot_VIT.png')
plt.close()

errors = np.array(denormalized_values_list_target) - np.array(denormalized_values_list_pred)
plt.scatter(range(len(errors)), errors, alpha=0.5)
plt.title('Error Plot: Prediction Error for Each Example')
plt.xlabel('Image')
plt.ylabel('Prediction Error')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Plot/errorplot_VIT.png')
plt.close()


plt.plot(rmse_list_val, label='RMSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) on Validation Set')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/Plot/rmse_val_plot_VIT.png')
plt.close()