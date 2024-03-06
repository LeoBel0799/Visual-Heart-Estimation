from facenet_pytorch import *
import neurokit2 as nk
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import glob
from scipy.stats import pearsonr
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
from PIL import Image

warnings.filterwarnings("ignore")


def denormalize(y, max_v, min_v):
    final_value = y * (max_v - min_v) + min_v
    return final_value


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                        :,
                        i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size,
                        ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=None):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"[x] - E: Can't divide dimension {d} into {n_heads} heads"

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

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
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

        assert chw[1] % n_patches == 0, "[x] - E: Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "[x] - E: Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, hidden_d)
        self.class_token = nn.Parameter(torch.randn(1, hidden_d))
        self.register_buffer("positional_embeddings", get_positional_embeddings(n_patches ** 2 + 1, hidden_d),
                             persistent=False)
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        self.ffnn = nn.Linear(hidden_d, out_d)

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            out = block(out)
        out = out[:, 0]
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


main_directory = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/video"
dataset_path = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/EVMCNN/EVMCNN_new.pth"

loaded_dataset = torch.load(dataset_path)
print("[INFO] - Global custom dataset loaded...!")

path_file = "/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/EVMCNN/min_max_values_EVMCNN.txt"

#
#
with open(path_file, 'r') as file:
    lines = file.readlines()
#
min_val_line = lines[0].strip().split(': ')
max_val_line = lines[1].strip().split(': ')

min_hr = float(min_val_line[1])
max_hr = float(max_val_line[1])


print(f'min_val: {min_hr}')
print(f'max_val: {max_hr}')



train_size = int(0.7 * len(loaded_dataset))
val_size = int(0.1 * len(loaded_dataset))
test_size = len(loaded_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(loaded_dataset, [train_size, val_size, test_size])

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



total_samples_in_train_loader = len(train_loader.dataset)
print(f"[INFO] - Total number of samples in train_loader: {total_samples_in_train_loader}")

total_samples_in_val_loader = len(val_loader.dataset)
print(f"[INFO] - Total number of samples in val_loader: {total_samples_in_val_loader}")

total_samples_in_test_loader = len(test_loader.dataset)
print(f"[INFO] - Total number of samples in test_loader: {total_samples_in_test_loader}")

# Creazione del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device,f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",)

criterion = nn.MSELoss()



def objective(trial, train, val):
    best_val_rmse = float('inf')
    LR = trial.suggest_float('LR', 1e-5, 1e-1, log=True)
    WD = trial.suggest_float('WD', 1e-5, 1e-1, log=True)
    PATIENCE = trial.suggest_int('PATIENCE', 2,3)
    num_epochs = trial.suggest_int('num_epochs', 4,5)
    val_loss_list = []
    n_patches = trial.suggest_categorical('n_patches', [1, 2, 4, 8])
    n_heads = trial.suggest_categorical('n_heads', [1,2,4])
    hidden_d = trial.suggest_categorical('hidden_d', [4,8,12])
    n_blocks = trial.suggest_int('n_block', 2,8)

    print(f"[INFO] - Trying parameters: patch_size={n_patches}, heads={n_heads}, hidden_d={hidden_d}, n_blocks={n_blocks}")
    print(f"[INFO] - Trying parameters: LR={LR}, WD={WD}, Patience={PATIENCE}, num of epochs={num_epochs}")


    model = MyViT((3,40,40), n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d,n_heads=n_heads)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()


    N_EPOCHS = num_epochs
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
                rmse_loss = torch.sqrt(loss)
            optimizer.zero_grad()
            scaler.scale(rmse_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {rmse_loss:.2f}")


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
                loss = torch.sqrt(criterion(outputs, targets_norm))
                val_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets_all.extend(targets_norm.cpu().numpy())

            mse = mean_squared_error(targets_all, predictions)
            rmse_validation = np.sqrt(mse)

            print(f"Epoch {epoch + 1}/{N_EPOCHS}, Validation RMSE: {rmse_validation:.4f}")
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
study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=10)
best_params = study.best_params
print("[INFO] - Best Hyperparameters:", best_params)
best_patch_size = best_params['n_patches']
best_heads = best_params['n_heads']
best_hidden = best_params['hidden_d']
best_nblock = best_params['n_block']
best_numepochs = best_params['num_epochs']
best_lr = best_params['LR']
best_wd= best_params['WD']
best_patience = best_params['PATIENCE']

best_model = MyViT((3,40,40), n_patches=best_patch_size, n_blocks=best_nblock, hidden_d=best_hidden,  n_heads=best_heads)
print(best_model)

LR = best_lr
PATIENCE = 10
WD = best_wd
optimizer = AdamW(best_model.parameters(), lr=LR, weight_decay=WD)
num_epochs = 100

best_model.to(device)

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

for epoch in range(num_epochs):
    best_model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, targets_norm, targets_original = batch
        images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(device)

        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)
        targets_original = targets_original.to(torch.float32)

        outputs = best_model(images)
        loss = criterion(outputs, targets_norm)
        rmse_loss = torch.sqrt(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} loss: {rmse_loss:.2f}")

    train_loss_list.append(rmse_loss)

    best_model.eval()
    predictions = []
    targets_all = []
    with torch.no_grad():
        val_loss = 0.0
        correct, total = 0, 0

        for batch in tqdm(val_loader, desc='Validation'):
            images, targets_norm, targets_original = batch
            images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(
                device)

            images = images.to(torch.float32)
            targets_norm = targets_norm.to(torch.float32)
            targets_original = targets_original.to(torch.float32)

            outputs = best_model(images)
            loss = torch.sqrt(criterion(outputs, targets_norm))
            predictions.extend(outputs.cpu().numpy())
            targets_all.extend(targets_norm.cpu().numpy())

    mse = mean_squared_error(targets_all, predictions)
    rmse_validation = np.sqrt(mse)
    rmse_list_val.append(rmse_validation)

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation RMSE: {rmse_validation:.4f}")
    # Early stopping check
    if rmse_validation < best_val_rmse:
        best_val_rmse = rmse_validation
        no_improvement_count = 0
    else:
        no_improvement_count += 1
        if no_improvement_count >= PATIENCE:
            print(f"[INFO] - No improvement for {PATIENCE} epochs. Early stopping.")
            break


torch.save(best_model.state_dict(),'/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/VIT_testrain.pt')

###############################################################################
# Test on train
model =  MyViT((3,40,40), n_patches=best_patch_size, n_blocks=best_nblock, hidden_d=best_hidden,  n_heads=best_heads)
model.load_state_dict(torch.load('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/VIT_testrain.pt', map_location='cpu'))
model.to('cuda')

predictions = []
targets_all = []
predictions_test = []
targets_all_test = []
test_loss_list = []

normalized_value_list = []
train_denormalized_values_list_pred = []
train_denormalized_values_list_target = []

predictions_train = []
targets_train = []
with torch.no_grad():
    test_loss = 0.0
    for batch in tqdm(train_loader, desc='Training'):
        images, targets_norm, targets_original = batch
        images, targets_norm, targets_original = images.to(device), targets_norm.to(device), targets_original.to(device)

        images = images.to(torch.float32)
        targets_norm = targets_norm.to(torch.float32)
        targets_original = targets_original.to(torch.float32)

        outputs = model(images)
        loss = torch.sqrt(criterion(outputs, targets_norm))
        test_loss += loss.item()
        predictions_train.extend(outputs.cpu().numpy())
        targets_train.extend(targets_norm.cpu().numpy())

    for value in predictions_train:
        denormalized_value = np.round(denormalize(value[0], max_hr,min_hr), 2)
        train_denormalized_values_list_pred.append(denormalized_value)

    for value in targets_train:
        denormalized_value = np.round(denormalize(value,max_hr,min_hr), 2)
        train_denormalized_values_list_target.append(denormalized_value)


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
correlation_coef = np.corrcoef(targets_all_array_train, predictions_array_train)[1,0]
r_squared = r2_score(targets_all_array_train, predictions_array_train)

print("\n--------------------------TRAINING METRICS--------------------------------\n")
print(f"RMSE: {rmse:.4f}) MAE: {mae:.2f}, MAPE: {mape:.2f}, R: {correlation_coef:.2f}, r^2: {r_squared:.2f}, Standard Deviation of Error (SDe): {sde:.2f}")


plt.scatter(targets_train, predictions_train, alpha=0.5, label='Predictions', color='red')
plt.scatter(targets_train, targets_train, alpha=0.5, label='Ground Truth', color='green')
plt.title('Scatter Plot: Ground Truth vs Prediction')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/scatterplot_VIT_train.png')
plt.close()

errors = np.array(train_denormalized_values_list_target) - np.array(train_denormalized_values_list_pred)
plt.scatter(range(len(errors)), errors, alpha=0.5)
plt.title('Error Plot: Prediction Error for Each Example')
plt.xlabel('Image')
plt.ylabel('Prediction Error')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/errorplot_VIT_train.png')
plt.close()

################################################################################
# Testing
model.eval()
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

        outputs = model(images)
        loss = torch.sqrt(criterion(outputs, targets_norm))
        test_loss += loss.item()
        predictions.extend(outputs.cpu().numpy())
        targets_all.extend(targets_norm.cpu().numpy())

    for value in predictions:
        denormalized_value = np.round(denormalize(value[0], max_hr,min_hr), 2)
        denormalized_values_list_pred.append(denormalized_value)

    for value in targets_all:
        denormalized_value = np.round(denormalize(value,max_hr,min_hr), 2)
        denormalized_values_list_target.append(denormalized_value)


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
correlation_coef = np.corrcoef(targets_all_array, predictions_array_test)[1,0]
r_squared = r2_score(targets_all_array, predictions_array_test)

print("\n--------------------------TEST METRICS--------------------------------\n")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}, R: {correlation_coef:.2f}, r^2: {r_squared}, Standard Deviation of Error (SDe): {sde:.2f}")

print("[INFO] - Best Hyperparameters for ViT:", best_params)


with open("/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/output.txt", "w") as file:
    file.write(f"Ground Truth: {denormalized_values_list_target}\n")
    file.write(f"Prediction: {denormalized_values_list_pred}\n")


plt.plot(denormalized_values_list_target, label='HR Original', marker='o')
plt.plot(denormalized_values_list_pred, label='Predictions', marker='x')
plt.title('True vs Predicted Values')
plt.ylabel('Predictions')
plt.legend()
plt.xticks([])
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/true_vs_predicted_VIT.png')
plt.close()

plt.plot(train_loss_list, label='Train Loss')
plt.plot(rmse_list_val, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/train_val_loss_VIT.png')
plt.close()


plt.scatter(targets_all, predictions, alpha=0.5, label='Predictions', color='red')
plt.scatter(targets_all, targets_all, alpha=0.5, label='Ground Truth', color='green')
plt.title('Scatter Plot: Ground Truth vs Prediction')
plt.xlabel('Ground Truth')
plt.ylabel('Prediction')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/scatterplot_VIT.png')
plt.close()

errors = np.array(denormalized_values_list_target) - np.array(denormalized_values_list_pred)
plt.scatter(range(len(errors)), errors, alpha=0.5)
plt.title('Error Plot: Prediction Error for Each Example')
plt.xlabel('Image')
plt.ylabel('Prediction Error')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/errorplot_VIT.png')
plt.close()


plt.plot(rmse_list_val, label='RMSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) on Validation Set')
plt.savefig('/home/ubuntu/data/ecg-fitness_raw-v1.0/dlib/VIT/rmse_val_plot_VIT.png')
plt.close()

