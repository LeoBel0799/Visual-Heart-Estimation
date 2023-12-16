import math
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
import math

"""# REGRESSIONE (mia)"""

# Carica i dati
data = pd.read_csv('/home/ubuntu/ecg-fitness_raw-v1.0/dataset_cleaned.csv')

# Rimuovi le righe con ECG HR negativo
data = data[data[' ECG HR'] >= 0]
data = data[data[' ECG HR'] >= 30]
# Estrai la colonna target
target_column = " ECG HR"  # Sostituisci con il nome effettivo della colonna target
target  = data[target_column]
target_next  = data[target_column]

# Calcola la correlazione
correlation_matrix = data.corr()

# Estrai la correlazione con la colonna target
correlation_with_target = correlation_matrix[target_column].drop(target_column)

# Puoi ordinare le colonne per correlazione decrescente
sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

# Visualizza tutte le righe
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(sorted_correlation)


# Estrai tutte le colonne tranne la colonna target
features = data.drop(columns=[target_column])

print(features)

colonne_da_elim = [col for col in features.columns if col.startswith('Landmark')]

# Elimina le colonne trovate
features = features.drop(columns=colonne_da_elim)
features = features.drop(columns=[' PPG', ' PPG HR', ' SpO2', ' PI'])
print(features)
features.to_csv('/home/ubuntu/ecg-fitness_raw-v1.0/elaborated_dataset.csv', index=False)

print(len(target))

print(list(features.columns))

print(features.head())


features = pd.read_csv('/home/ubuntu/ecg-fitness_raw-v1.0/elaborated_dataset.csv')
# Estrai tutte le colonne tranne la colonna target
numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns

# Crea un oggetto StandardScaler
scaler = StandardScaler()

# Applica la Z-Score Normalization alle colonne numeriche di features
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

# Normalizza la colonna target separatamente
target = data[target_column]
normalized_target = scaler.fit_transform(target.values.reshape(-1, 1))

sample_size = 100
sampled_target = target.head(sample_size)
sampled_normalized_target = normalized_target[:sample_size]

# Crea un'istanza di figura e assi
fig, ax = plt.subplots(figsize=(10, 6))

# Plot dei valori originali in background (blu chiaro)
ax.bar(sampled_target.index, sampled_target, color='lightblue', label='Original Values')

# Plot dei valori normalizzati in primo piano (blu scuro)
ax.bar(sampled_target.index, sampled_normalized_target.flatten(), color='darkblue', label='Normalized Values')

# Aggiungi etichette e titolo
ax.set_xlabel('Indice')
ax.set_ylabel('Valori')
ax.set_title('Confronto tra Valori Originali e Normalizzati nella Colonna Target (Primi 100 Elementi)')
ax.legend()

# Mostra il plot
plt.show()

X_train, X_temp, y_train, y_temp = train_test_split(features, normalized_target, test_size=0.2, random_state=42)
print("x train:",type(X_train))
print("x_temp:",type(X_temp))
print("y_train:",type(y_train))
print("y_temp:",type(y_temp))
# Create test and validation sets
feature_names = X_train.columns.tolist()

X_train_np = X_train.values
X_temp_np = X_temp.values

# Suddividi il set temporaneo in test e validation
X_test, X_val, y_test, y_val = train_test_split(X_temp_np, y_temp, test_size=0.5, random_state=42)

# Converti NumPy arrays in tensori PyTorch
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Crea TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
val_dataset = TensorDataset(X_val, y_val)

plt.bar(*np.unique(y_train, return_counts=True))
plt.title('Distribution of Classes in Training Resampled Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Plotta la distribuzione delle classi per il set di test
plt.bar(*np.unique(y_test, return_counts=True))
plt.title('Distribution of Classes in Test Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Plotta la distribuzione delle classi per il set di validazione
plt.bar(*np.unique(y_val, return_counts=True))
plt.title('Distribution of Classes in Validation Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

"""ADDESTRAMENTO E APPLICAZIONE SUCCESSIVA DI CAPTUM PER COMPRENDERE FEATURE IMPORTANCE"""

class DeeperPhysioLandmarkHRPredictor(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, dropout_rate):
        super(DeeperPhysioLandmarkHRPredictor, self).__init__()

        # Linear layers
        self.fc_input = nn.Linear(input_size, hidden1_size)
        self.fc_hidden1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc_hidden2 = nn.Linear(hidden2_size, hidden3_size)
        self.fc_hidden3 = nn.Linear(hidden3_size, hidden4_size)
        self.fc_output = nn.Linear(hidden4_size, 1)

        # Batch Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(hidden1_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden2_size)
        self.batch_norm3 = nn.BatchNorm1d(hidden3_size)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Inizializzazione Xavier/Glorot
        nn.init.xavier_uniform_(self.fc_input.weight)
        nn.init.xavier_uniform_(self.fc_hidden1.weight)
        nn.init.xavier_uniform_(self.fc_hidden2.weight)
        nn.init.xavier_uniform_(self.fc_hidden3.weight)
        nn.init.xavier_uniform_(self.fc_output.weight)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc_input(x)))
        x = self.dropout1(x)
        x = F.relu(self.batch_norm2(self.fc_hidden1(x)))
        x = self.dropout2(x)
        x = F.relu(self.batch_norm3(self.fc_hidden2(x)))
        x = self.dropout3(x)
        x = F.relu(self.fc_hidden3(x))
        hr_prediction = self.fc_output(x)

        return hr_prediction

results = []
# Funzione obiettivo da minimizzare (in questo caso, la loss sul validation set)
def objective(params):
    # Aggiornamento dei parametri del modello con quelli ottimizzati
    hidden1_size = int(params['hidden1_size'])
    hidden2_size = int(params['hidden2_size'])
    hidden3_size = int(params['hidden3_size'])
    hidden4_size = int(params['hidden4_size'])
    dropout_rate = params['dropout_rate']
    print(f"\nOptimizing with parameters: hidden1_size={hidden1_size}, hidden2_size={hidden2_size}, hidden3_size={hidden3_size} hidden4_size={hidden4_size}dropout_rate={dropout_rate}, "
          f"learning_rate={params['lr']}, batch_size={int(params['batch_size'])}\n")

    model = DeeperPhysioLandmarkHRPredictor(features.shape[1], hidden1_size, hidden2_size,hidden3_size,hidden4_size,dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
    trainloader = DataLoader(train_dataset, batch_size=int(params['batch_size']), shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=int(params['batch_size']), shuffle=False)

    num_epochs = 30
    early_stopping_patience = 8
    best_validation_loss = float('inf')
    patience_counter = 0
    criterion = nn.MSELoss()  # o qualsiasi altra funzione di loss necessaria per il tuo problema
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                validation_loss += loss.item()

        # Early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    print(f"\nBest parameters: hidden1_size={hidden1_size},hidden2_size={hidden2_size},hidden3_size={hidden3_size},hidden4_size={hidden4_size} dropout_rate={dropout_rate}, "
          f"learning_rate={params['lr']}, batch_size={int(params['batch_size'])}, "
          f"Best Validation Loss: {best_validation_loss}\n")

    # Aggiungi i risultati alla lista
    results.append({
        'lr': params['lr'],
        'batch_size': int(params['batch_size']),
        'hidden1_size': int(params['hidden1_size']),
        'hidden2_size': int(params['hidden2_size']),
        'hidden3_size': int(params['hidden3_size']),
        'hidden4_size': int(params['hidden4_size']),
        'dropout_rate': params['dropout_rate'],
        'validation_loss': best_validation_loss
    })


    return best_validation_loss

# Definizione dello spazio di ricerca
space = {
    'lr': hp.loguniform('lr', -5, 0),  # Search in the log scale from 1e-5 to 1
    'batch_size': hp.quniform('batch_size', 32, 128, 1),  # Discrete values from 32 to 128
    'hidden1_size': hp.quniform('hidden1_size', 32, 64, 1),
    'hidden2_size': hp.quniform('hidden2_size', 64, 128, 1),
    'hidden3_size': hp.quniform('hidden3_size', 32, 64, 1),
    'hidden4_size': hp.quniform('hidden4_size', 10,32, 1),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
}
# Esecuzione dell'ottimizzazione bayesiana
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20)
# Stampa i risultati ottenuti
print(results)


# Stampa dei migliori hyperparameters trovati
best_learning_rate = best['lr']
best_batch_size = int(best['batch_size'])
best_hidden1_size = int(best['hidden1_size'])
best_hidden2_size = int(best['hidden2_size'])
best_hidden3_size = int(best['hidden3_size'])
best_hidden4_size = int(best['hidden4_size'])
best_dropout_rate = best['dropout_rate']

print(f'Best Learning Rate: {best_learning_rate}, Best Batch Size: {best_batch_size}')
print(f'Best Hidden1 Size: {best_hidden1_size}, Best Hidden2 Size: {best_hidden2_size}, Best Hidden3 Size: {best_hidden3_size}, Best Hidden4 Size: {best_hidden4_size}')
print(f'Best Dropout Rate: {best_dropout_rate}')

trainloader = DataLoader(train_dataset, batch_size=int(best['batch_size']), shuffle=True)
valloader = DataLoader(val_dataset, batch_size=int(best['batch_size']), shuffle=False)
testloader = DataLoader(test_dataset,  batch_size=int(best['batch_size']), shuffle=False)


early_stopping_patience = 5
best_rmse = float('inf')
best_epoch = 0

train_rmse_list = []
val_rmse_list = []
train_loss_list = []
val_loss_list = []
training_losses = []
validation_losses = []

# Addestramento finale con i migliori hyperparameters
final_model = DeeperPhysioLandmarkHRPredictor(features.shape[1], best_hidden1_size, best_hidden2_size, best_hidden3_size, best_hidden4_size, best_dropout_rate)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_learning_rate, weight_decay=1e-5)

final_trainloader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
final_valloader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)  # Add this line if not defined

criterion = nn.MSELoss()  # Mean Squared Error loss
ig = IntegratedGradients(final_model)

# Spostamento del modello e dell'ottimizzatore sulla GPU, se disponibile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model.to(device)

num_epochs = 25  # Aumenta il numero di epoche per l'addestramento finale
for epoch in range(num_epochs):
    final_model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for i, data in enumerate(final_trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        final_optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        final_optimizer.step()

        running_loss += loss.item()
        all_predictions.extend(outputs.tolist())
        all_labels.extend(labels.tolist())

    mse = mean_squared_error(all_labels, all_predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_labels, all_predictions)

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(final_trainloader)}, RMSE: {rmse}, R²: {r2}')
    training_losses.append(running_loss / len(final_trainloader))
    train_rmse_list.append(rmse)

    # Validation
    final_model.eval()
    val_running_loss = 0.0
    val_all_predictions = []
    val_all_labels = []

    with torch.no_grad():
        for val_data in final_valloader:
            val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = final_model(val_inputs)
            val_loss = criterion(val_outputs, val_labels.view(-1, 1))

            val_running_loss += val_loss.item()
            val_all_predictions.extend(val_outputs.tolist())
            val_all_labels.extend(val_labels.tolist())

    val_mse = mean_squared_error(val_all_labels, val_all_predictions)
    val_rmse = math.sqrt(val_mse)
    val_r2 = r2_score(val_all_labels, val_all_predictions)

    print(f'Validation Loss: {val_running_loss / len(final_valloader)}, Validation RMSE: {val_rmse}, Validation R²: {val_r2}')
    validation_losses.append(val_running_loss / len(final_valloader))
    val_rmse_list.append(val_rmse)
    val_loss_list.append(val_running_loss / len(final_valloader))

    # Check for early stopping
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_epoch = epoch
    elif epoch - best_epoch >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch+1}. Best RMSE: {best_rmse}')
        break

# Calcola le attribuzioni di Captum alla fine dell'addestramento
final_model.eval()
all_attributions = []

with torch.no_grad():
    for data in final_trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        attributions, _ = ig.attribute(inputs,return_convergence_delta=True)
        all_attributions.append(attributions.detach().numpy())

feature_importance = np.concatenate(all_attributions, axis=0).mean(axis=0)



final_model.eval()
test_loss = 0.0
all_test_predictions = []
all_test_labels = []

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = final_model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        test_loss += loss.item()

        all_test_predictions.extend(outputs.tolist())
        all_test_labels.extend(labels.tolist())

test_mse = mean_squared_error(all_test_labels, all_test_predictions)
test_rmse = math.sqrt(test_mse)
test_r2 = r2_score(all_test_labels, all_test_predictions)

print(f'Test Loss: {test_loss / len(testloader)}')
print(f'Test RMSE: {test_rmse}, Test R²: {test_r2}')

print(f'Test Loss: {test_loss / len(testloader)}')
print(f'Test RMSE: {test_rmse}, Test R²: {test_r2}')

all_test_labels_original = scaler.inverse_transform(np.array(all_test_labels).reshape(-1, 1)).flatten()
all_test_predictions_original = scaler.inverse_transform(np.array(all_test_predictions).reshape(-1, 1)).flatten()
torch.save(final_model, '/home/ubuntu/ecg-fitness_raw-v1.0/MineRegressor.pth')


plt.plot(all_test_labels, label='Ground Truth')
plt.plot(all_test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.title('Test Set: Predicted vs Ground Truth Over Time')
plt.show()


# Creazione di un nuovo grafico a dispersione
plt.figure(figsize=(10, 6))
plt.scatter(all_test_labels_original, all_test_predictions_original, alpha=0.5)
plt.plot([min(all_test_labels_original), max(all_test_labels_original)],
         [min(all_test_labels_original), max(all_test_labels_original)],
         color='red', linestyle='--', linewidth=2)
plt.title('Test Set Predictions vs. Actual Labels (Original Scale)')
plt.xlabel('Actual Labels (Original Scale)')
plt.ylabel('Predicted Labels (Original Scale)')
plt.show()
min_length = min(len(train_loss_list), len(val_loss_list))
train_loss_list = train_loss_list[:min_length]
val_loss_list = val_loss_list[:min_length]

plt.plot(training_losses, label='Train Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Creazione di un grafico a barre per visualizzare l'importanza delle feature
colors = ['green' if importance >= 0 else 'red' for importance in feature_importance]

plt.figure(figsize=(10,6))  # Aumenta la dimensione della figura
plt.bar(feature_names, feature_importance, color=colors, width=0.2, linewidth=2)  # Aumenta l'ampiezza delle colonne e l'ampiezza del bordo
plt.title('Feature Importance', fontsize=20)  # Imposta la dimensione del titolo
plt.xlabel('Feature Name', fontsize=20)  # Imposta la dimensione dell'etichetta x
plt.ylabel('Importance', fontsize=20)  # Imposta la dimensione dell'etichetta y
plt.xticks(rotation=90, fontsize=10)  # Ruota i nomi delle feature e imposta la dimensione del carattere
plt.subplots_adjust(right=0.7)  # Regola il margine destro per estendere la lunghezza dei nomi delle feature
plt.tight_layout()  # Ottimizza la disposizione della figura
plt.show()

