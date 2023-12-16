import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
from decimal import *



"""# DATA PREPROCESSING, UNIONE DEI DATI FISIOLIGICI E FACCIALI + ESTRAZIONE FEATURE DA LANDMARK FACCIALI PER HR ESTIMATION (REGRESSION)"""

def process_ecg_data_all(main_directory,fps=30):
    # Crea una lista per memorizzare i dati della seconda colonna di tutti i file CSV

    # Itera su tutte le cartelle principali
    for root, dirs, files in os.walk(main_directory):
        # Verifica se stiamo processando una cartella principale
            for file in files:
                if file == "viatom-raw.csv":
                    # Ottieni il percorso completo del file CSV
                    file_path = os.path.join(root, file)
                    # Carica il CSV in un DataFrame
                    df = pd.read_csv(file_path, na_values=['NA', 'N/A', 'NaN'])
                    df['milliseconds'] = (df['milliseconds'] - df['milliseconds'].iloc[0])

                    # Rimuovi i duplicati nella colonna 'milliseconds'
                    df = df.drop_duplicates(subset=['milliseconds'])

                    # Calcola la differenza tra il primo e l'ultimo valore in millisecondi
                    millisecondi_diff = df['milliseconds'].iloc[-1] - df['milliseconds'].iloc[0]

                    # Definisci un valore massimo in millisecondi per un minuto
                    millisecondi_al_minuto = 60000

                    # Se la differenza tra il primo e l'ultimo valore supera un minuto
                    if millisecondi_diff > millisecondi_al_minuto:
                        # Calcola il tempo inizio e fine per un minuto
                        inizio_millisecondi = df['milliseconds'].iloc[0]
                        fine_millisecondi = inizio_millisecondi + millisecondi_al_minuto

                        # Filtra il DataFrame in modo da mantenere solo i dati di un minuto
                        df = df[(df['milliseconds'] >= inizio_millisecondi) & (df['milliseconds'] <= fine_millisecondi)]

                    # Calcola il passo di campionamento per l'ECG in base al frame rate del video
                    ecg_sampling_interval = 1.0 / fps

                    # Crea un array di timestamp per i dati ECG campionati
                    ecg_timestamps = np.arange(0, len(df) * ecg_sampling_interval, ecg_sampling_interval)
                    ecg_timestamps = (ecg_timestamps * 1000).astype(int)

                    # Campiona i dati dell'ECG in base ai timestamp
                    sampled_ecg_data = df.copy()

                    sampled_ecg_data['milliseconds'] = ecg_timestamps

                    folder_name = os.path.basename(root)
                    ecg_60_seconds_file = os.path.join(root, f"ecg_60_seconds_{folder_name}.csv")
                    sampled_ecg_data.to_csv(ecg_60_seconds_file, index=False)


def create_or_recreate_file(features_file_path, feature_names):
    # Numero massimo di tentativi di creazione del file
    max_retries = 3
    retries = 0

    while retries < max_retries:
        try:
            with open(features_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(feature_names)
            return True
        except Exception as e:
            print(f"Errore durante la creazione del file: {str(e)}")
            retries += 1

    return False

def calculate_lips_proportions(landmarks):
    # Lip landmarks indices
    upper_lip_start_index = 48
    upper_lip_end_index = 54
    lower_lip_start_index = 51
    lower_lip_end_index = 57
    middle_point_index = 62

    # Extract coordinates of lip landmarks
    upper_lip_start = (landmarks[upper_lip_start_index].x, landmarks[upper_lip_start_index].y)
    upper_lip_end = (landmarks[upper_lip_end_index].x, landmarks[upper_lip_end_index].y)
    lower_lip_start = (landmarks[lower_lip_start_index].x, landmarks[lower_lip_start_index].y)
    lower_lip_end = (landmarks[lower_lip_end_index].x, landmarks[lower_lip_end_index].y)
    middle_point = (landmarks[middle_point_index].x, landmarks[middle_point_index].y)

    # Calculate lips width and height
    lips_width = upper_lip_end[0] - upper_lip_start[0]
    lips_height = (lower_lip_start[1] + lower_lip_end[1]) / 2 - middle_point[1]

    # Calculate face width
    face_width = landmarks[16].x - landmarks[0].x

    # Calculate lips width and height ratios
    lips_width_ratio = lips_width / face_width
    lips_height_ratio = lips_height / face_width

    return lips_width_ratio, lips_height_ratio

def calculate_eyebrow_height_to_eye(landmarks):
    # Extract coordinates of left and right eyebrow landmarks
    left_eyebrow_landmarks = [(point.x, point.y) for point in landmarks[17:22]]
    right_eyebrow_landmarks = [(point.x, point.y) for point in landmarks[22:27]]

    # Convert the lists to NumPy arrays
    left_eyebrow = np.array(left_eyebrow_landmarks)
    right_eyebrow = np.array(right_eyebrow_landmarks)

    # Calculate eyebrow heights
    height_left_eyebrow = left_eyebrow[0][1] - left_eyebrow[4][1]
    height_right_eyebrow = right_eyebrow[0][1] - right_eyebrow[4][1]

    # Calculate average eyebrow height
    average_eyebrow_height = (height_left_eyebrow + height_right_eyebrow) / 2

    return average_eyebrow_height

def calculate_eye_opening(landmarks):
    # Extract coordinates of left eye landmarks
    left_eye = [(point.x, point.y) for point in landmarks[42:48]]

    # Extract coordinates of right eye landmarks
    right_eye = [(point.x, point.y) for point in landmarks[36:42]]

    # Convert the lists to NumPy arrays
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)

    # Calculate eye opening for left and right eyes
    eye_opening_left = left_eye[1][1] - left_eye[4][1]
    eye_opening_right = right_eye[1][1] - right_eye[4][1]

    # Calculate average eye opening
    average_eye_opening = (eye_opening_left + eye_opening_right) / 2

    return average_eye_opening

def calculate_color_intensity_on_cheeks(frame, landmarks):
    # Considera la regione delle guance tra le landmarks 0 e 16 per il calcolo dell'intensità del colore
    cheek_region = np.array([(point.x, point.y) for point in landmarks[0:17]])

    # Estrai le coordinate x e y delle guance
    cheek_x = [point[0] for point in cheek_region]
    cheek_y = [point[1] for point in cheek_region]
    cheek_y = np.clip(cheek_y, 0, frame.shape[0] - 1)
    cheek_x = np.clip(cheek_x, 0, frame.shape[1] - 1)


    # Calcola la media dell'intensità del colore nelle guance per ogni canale di colore (RGB)
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [cheek_region.astype(np.int32)], color=(255, 255, 255))

    # Estrai i valori dei canali di colore nelle guance
    cheek_colors = frame[cheek_y, cheek_x]

    # Calcola la media per ciascun canale di colore
    cheek_color_intensity = np.mean(cheek_colors, axis=0)

    # Restituisci i valori medi dei canali di colore separatamente per rosso, verde e blu
    average_red_intensity = cheek_color_intensity[0]
    average_green_intensity = cheek_color_intensity[1]
    average_blue_intensity = cheek_color_intensity[2]

    return average_red_intensity, average_green_intensity, average_blue_intensity


def calculate_facial_expressions(landmarks):
    left_eye = np.array([(point.x, point.y) for point in landmarks[42:48]])
    right_eye = np.array([(point.x, point.y) for point in landmarks[36:42]])
    mouth = np.array([(point.x, point.y) for point in landmarks[48:68]])

    eye_aspect_ratio = (np.linalg.norm(left_eye[1] - left_eye[5]) + np.linalg.norm(left_eye[2] - left_eye[4])) / (2.0 * np.linalg.norm(left_eye[0] - left_eye[3]))
    mouth_aspect_ratio = (np.linalg.norm(mouth[1] - mouth[7]) + np.linalg.norm(mouth[2] - mouth[6]) + np.linalg.norm(mouth[3] - mouth[5])) / (3.0 * np.linalg.norm(mouth[0] - mouth[4]))

    return eye_aspect_ratio, mouth_aspect_ratio

def calculate_facial_proportions(landmarks):
    # Calcola proporzioni del viso, ad esempio, larghezza occhi / larghezza viso
    left_eye = np.array([(point.x, point.y) for point in landmarks[42:48]])
    right_eye = np.array([(point.x, point.y) for point in landmarks[36:42]])

    width_face = np.linalg.norm((landmarks[16].x - landmarks[0].x, landmarks[16].y - landmarks[0].y))  # Distanza tra le guance
    width_eyes = np.linalg.norm(left_eye[0] - left_eye[3]) + np.linalg.norm(right_eye[0] - right_eye[3])  # Somma delle larghezze degli occhi

    face_width_ratio = width_eyes / width_face

    return face_width_ratio

def calculate_eyebrow_features(landmarks):
    # Calcola caratteristiche delle sopracciglia, ad esempio, altezza media delle sopracciglia
    left_eyebrow = np.array([(point.x, point.y) for point in landmarks[17:22]])
    right_eyebrow = np.array([(point.x, point.y) for point in landmarks[22:27]])

    height_left_eyebrow = (left_eyebrow[4][1] - left_eyebrow[0][1]) / np.linalg.norm(left_eyebrow[0] - left_eyebrow[4])
    height_right_eyebrow = (right_eyebrow[4][1] - right_eyebrow[0][1]) / np.linalg.norm(right_eyebrow[0] - right_eyebrow[4])

    average_eyebrow_height = (height_left_eyebrow + height_right_eyebrow) / 2

    return average_eyebrow_height


def calculate_front_color_intensity(frame, landmarks):
    # Ad esempio, considera la regione della fronte tra le landmarks 17 e 26 per il calcolo dell'intensità del colore
    forehead_region = np.array([(point.x, point.y) for point in landmarks[17:27]])

    # Estrai le coordinate x e y della regione della fronte
    forehead_x = [point[0] for point in forehead_region]
    forehead_y = [point[1] for point in forehead_region]
    forehead_y = np.clip(forehead_y, 0, frame.shape[0] - 1)
    forehead_x = np.clip(forehead_x, 0, frame.shape[1] - 1)
    # Calcola la media dell'intensità del colore nella regione della fronte per ogni canale di colore (RGB)
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [forehead_region.astype(np.int32)], color=(255, 255, 255))

    # Estrai i valori dei canali di colore nella regione della fronte
    forehead_colors = frame[forehead_y, forehead_x]

    # Calcola la media per ciascun canale di colore
    forehead_color_intensity = np.mean(forehead_colors, axis=0)

    # Restituisci i valori medi dei canali di colore separatamente per rosso, verde e blu
    average_red_intensity = forehead_color_intensity[0]
    average_green_intensity = forehead_color_intensity[1]
    average_blue_intensity = forehead_color_intensity[2]

    return average_red_intensity, average_green_intensity, average_blue_intensity

def calculate_color_intensity_on_lips(frame, landmarks):
    # Considera la regione delle labbra tra le landmarks 48 e 68 per il calcolo dell'intensità del colore
    lips_region = np.array([(point.x, point.y) for point in landmarks[48:68]])

    # Estrai le coordinate x e y delle labbra
    lips_x = [point[0] for point in lips_region]
    lips_y = [point[1] for point in lips_region]
    lips_y = np.clip(lips_y, 0, frame.shape[0] - 1)
    lips_x = np.clip(lips_x, 0, frame.shape[1] - 1)

    # Calcola la media dell'intensità del colore sulle labbra per ogni canale di colore (RGB)
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [lips_region.astype(np.int32)], color=(255, 255, 255))

    # Estrai i valori dei canali di colore nelle labbra
    lips_colors = frame[lips_y, lips_x]

    # Calcola la media per ciascun canale di colore
    lips_color_intensity = np.mean(lips_colors, axis=0)

    # Restituisci i valori medi dei canali di colore separatamente per rosso, verde e blu
    average_red_intensity = lips_color_intensity[0]
    average_green_intensity = lips_color_intensity[1]
    average_blue_intensity = lips_color_intensity[2]

    return average_red_intensity, average_green_intensity, average_blue_intensity


def calculate_nose_proportions(landmarks):
    # Nose landmarks indices
    nose_tip_index = 30
    nose_bridge_start_index = 27
    nose_bridge_end_index = 35

    # Extract coordinates of nose landmarks
    nose_tip = (landmarks[nose_tip_index].x, landmarks[nose_tip_index].y)
    nose_bridge_start = (landmarks[nose_bridge_start_index].x, landmarks[nose_bridge_start_index].y)
    nose_bridge_end = (landmarks[nose_bridge_end_index].x, landmarks[nose_bridge_end_index].y)

    # Calculate nose width and height
    nose_width = nose_bridge_end[0] - nose_bridge_start[0]
    nose_height = nose_tip[1] - nose_bridge_start[1]

    # Calculate face width and height
    face_width = landmarks[16].x - landmarks[0].x
    face_height = landmarks[8].y - landmarks[27].y

    # Calculate nose width and height ratios
    nose_width_ratio = nose_width / face_width
    nose_height_ratio = nose_height / face_height

    return nose_width_ratio, nose_height_ratio


def calculate_power_spectrum(intensity_forehead, intensity_lips, intensity_cheeks):
    # Applica la FFT per la fronte
    fft_forehead = np.fft.fft(intensity_forehead)
    power_spectrum_forehead = np.abs(fft_forehead) ** 2
    frequencies_forehead = np.fft.fftfreq(len(intensity_forehead))

    # Applica la FFT per le labbra
    fft_lips = np.fft.fft(intensity_lips)
    power_spectrum_lips = np.abs(fft_lips) ** 2
    frequencies_lips = np.fft.fftfreq(len(intensity_lips))

    # Applica la FFT per le guance
    fft_cheeks = np.fft.fft(intensity_cheeks)
    power_spectrum_cheeks = np.abs(fft_cheeks) ** 2
    frequencies_cheeks = np.fft.fftfreq(len(intensity_cheeks))

    return (frequencies_forehead, power_spectrum_forehead), (frequencies_lips, power_spectrum_lips), (frequencies_cheeks, power_spectrum_cheeks)


def delete_old_files(main_directory):
    # Ottieni la data corrente
    current_date = datetime.now().date()

    for root, dirs, files in os.walk(main_directory):
        # Verifica se stiamo processando una cartella principale
        for file in files:
            if file.startswith("face"):
                file_path = os.path.join(root, file)

                try:
                    # Ottieni il timestamp dell'ultima modifica del file
                    last_modified_timestamp = os.path.getmtime(file_path)

                    # Converti il timestamp in una data
                    last_modified_date = datetime.fromtimestamp(last_modified_timestamp).date()

                    # Verifica se il file non è stato modificato oggi
                    if last_modified_date != current_date:
                        # Elimina il file se non è stato modificato oggi
                        os.remove(file_path)
                        print(f"Il file {file} è stato eliminato perché non è stato modificato oggi.")
                except Exception as e:
                    print(f"Errore durante la verifica del file {file}: {str(e)}")


def extract_facial_features_all(main_directory):
    # Inizializza il rilevatore di volti e il predittore delle landmark facciali
    face_detector = dlib.cnn_face_detection_model_v1('/home/ubuntu/ecg-fitness_raw-v1.0/mmod_human_face_detector.dat')  # Modello CNN per il rilevamento dei volti (compatibile con GPU)
    landmark_predictor = dlib.shape_predictor('/home/ubuntu/ecg-fitness_raw-v1.0/shape_predictor_68_face_landmarks_GTX.dat')  # Assicurati di fornire il percorso corretto al file dei landmark

    # Itera su tutte le cartelle principali
    for root, dirs, files in os.walk(main_directory):
        # Verifica se stiamo processando una cartella principale
        for file in files:
            if file == "c920-1.avi":
                # Apri il video
                file_path = os.path.join(root, file)
                video_capture = cv2.VideoCapture(file_path)

                # Inizializza una lista per salvare le feature facciali
                facial_features_list = []
                feature_names = ["milliseconds"]
                for i in range(68):
                    feature_names.append(f"Landmark_{i}")
                feature_names += ["EyeAspectRatio", "MouthAspectRatio", "LightIntensity", "FaceWidthRatio", "EyebrowHeight","LipsWidthRatio",
                                  "LipsHeightRatio","EyeOpening","NoseWidthRatio","NoseHeightRatio","AvgRedForehead","AvgGreenForehead","AvgBlueForehead",
                                  "AvgRedLips","AvgGreenLips","AvgBlueLips","AvgRedCheeks","AvgGreenCheeks","AvgBlueCheeks"]
                # Salva le feature facciali in un file di output
                folder_name = os.path.basename(root)
                features_file_name = f"face_feature_dlib_{folder_name}.csv"
                features_file_path = os.path.join(root, features_file_name)

                if os.path.exists(features_file_path):
                    print(f"File già esistente: {features_file_path}")
                    continue
                # Crea il file e verifica se è stato creato correttamente
                if not create_or_recreate_file(features_file_path, feature_names):
                    # Se la creazione non ha avuto successo, passa alla cartella successiva
                    continue
                # Utilizza tqdm per creare una barra di avanzamento per il numero totale di frame nel video
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                progress_bar = tqdm(total=total_frames, desc=f'Processing {file_path}', position=0, leave=True)


                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    # Rileva i volti nel frame utilizzando il modello CNN
                    faces = face_detector(frame, 0)

                    for face in faces:
                        landmarks = landmark_predictor(frame, face.rect)
                        facial_features = [landmarks.part(i) for i in range(68)]
                        eye_aspect_ratio, mouth_aspect_ratio = calculate_facial_expressions(facial_features)
                        light_intensity = int(cv2.mean(frame)[0])
                        face_width_ratio = calculate_facial_proportions(facial_features)
                        eyebrow_height = calculate_eyebrow_features(facial_features)
                        average_red_intensity_cheeks, average_green_intensity_cheeks, average_blue_intensity_cheeks = calculate_color_intensity_on_cheeks(frame,facial_features)
                        average_red_intensity_lips, average_green_intensity_lips, average_blue_intensity_lips = calculate_color_intensity_on_lips(frame,facial_features)
                        average_red_intensity_forehead, average_green_intensity_forehead, average_blue_intensity_forehead = calculate_front_color_intensity(frame,facial_features)
                        cheek_intensity = calculate_color_intensity_on_cheeks(frame, facial_features)
                        forehead_intensity = calculate_front_color_intensity(frame, facial_features)
                        lips_color_intensity = calculate_color_intensity_on_lips(frame, facial_features)
                        lips_width_ratio, lips_height_ratio = calculate_lips_proportions(facial_features)
                        average_eyebrow_height = calculate_eye_opening(facial_features)
                        nose_width_ratio, nose_height_ratio = calculate_nose_proportions(facial_features)
                        #intensity_forehead = np.array([average_red_intensity_forehead, average_green_intensity_forehead, average_blue_intensity_forehead])
                        #intensity_lips = np.array([average_red_intensity_lips, average_green_intensity_lips, average_blue_intensity_lips])
                        #intensity_cheeks = np.array([average_red_intensity_cheeks, average_green_intensity_cheeks, average_blue_intensity_cheeks])
                        #(frequencies_forehead, power_spectrum_forehead), (frequencies_lips, power_spectrum_lips), (frequencies_cheeks, power_spectrum_cheeks) = calculate_power_spectrum(intensity_forehead, intensity_lips, intensity_cheeks)

                        timestamp = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))

                        facial_features_with_timestamp = (
                            timestamp,
                            *[(point.x, point.y) for point in facial_features],
                            float(Decimal(str(eye_aspect_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(mouth_aspect_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(light_intensity)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(face_width_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(eyebrow_height)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(lips_width_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(lips_height_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_eyebrow_height)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(nose_width_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(nose_height_ratio)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_red_intensity_forehead)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_green_intensity_forehead)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_blue_intensity_forehead)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_red_intensity_lips)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_green_intensity_lips)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_blue_intensity_lips)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_red_intensity_cheeks)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_green_intensity_cheeks)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            float(Decimal(str(average_blue_intensity_cheeks)).quantize(Decimal('0.00'), rounding=ROUND_DOWN)),
                            #(frequencies_forehead, power_spectrum_forehead),
                            #(frequencies_lips, power_spectrum_lips),
                            #(frequencies_cheeks, power_spectrum_cheeks)
                        )
                        facial_features_list.append(facial_features_with_timestamp)
                        progress_bar.update(1)

                progress_bar.close()
                print(f"CSV for {file_path} has been written.")
                # Riapri il file in modalità "aggiunta" per aggiungere le nuove feature facciali
                with open(features_file_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for features in facial_features_list:
                        csv_writer.writerow(features)

                # Chiudi il video
                video_capture.release()

def merge_csv_files(main_directory, output_csv_filename):
    all_data = []  # Lista in cui verranno memorizzati i dati dai file CSV
    header = None  # Inizializza l'header come None

    # Itera su tutte le cartelle principali
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.startswith("final"):  # Assicurati che il file sia un file CSV
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                if header is None:
                    # Salva l'header dal primo file CSV trovato
                    header = df.columns.tolist()

                all_data.append(df)

    if all_data:
        # Rimuovi le righe in cui il valore di "ECG HR" è negativo
        merged_data = merged_data[merged_data[' ECG HR'] >= 0]

        # Salva il DataFrame unito in un nuovo file CSV nella directory principale
        output_csv_path = os.path.join(main_directory, output_csv_filename)
        merged_data.to_csv(output_csv_path, index=False, header=header)
        print(f"I file CSV sono stati uniti con successo in '{output_csv_filename}' nella directory principale.")
    else:
        print("Nessun file CSV trovato nelle sotto-cartelle.")

def merge_ecg_and_facial_features_landmarks(main_directory):
    # Itera su tutte le sottocartelle delle 16 cartelle principali
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.startswith("ecg"):
                folder_name = os.path.basename(root)
                raw_proc_ecg = os.path.join(root, file)

                # Carica il CSV dei dati ECG campionati
                ecg_df = pd.read_csv(raw_proc_ecg)

                facial_features = os.path.join(root, f"face_feature_dlib_{folder_name}.csv")
                facial_features_df = pd.read_csv(facial_features)

                # Unisci i due DataFrame in base al timestamp (milliseconds)
                merged_df = pd.merge(ecg_df, facial_features_df, on='milliseconds')
                print(merged_df)

                # Salva il DataFrame unito in un nuovo file CSV
                final_csv = os.path.join(root, f"final_{folder_name}.csv")
                merged_df.to_csv(final_csv, index=False)


main_directory = '/home/ubuntu/ecg-fitness_raw-v1.0'
#process_ecg_data_all(main_directory)
#delete_old_files(main_directory)
#extract_facial_features_all(main_directory)
#merge_ecg_and_facial_features_landmarks(main_directory)
#merge_csv_files(main_directory, "dataset.csv")


csv_file = "/home/ubuntu/dataset.csv"  # Sostituisci con il percorso del tuo file CSV
df = pd.read_csv(csv_file)

# Conta il numero totale di righe escludendo l'header
num_total_rows = len(df)

# Estrai la colonna HR (assicurati che il nome della colonna sia corretto)
hr_column = " ECG HR"  # Sostituisci con il nome effettivo della colonna HR

# Crea un DataFrame contenente solo la colonna HR
hr_df = df[[hr_column]]

# Conta i valori univoci nella colonna HR
unique_hr_values = hr_df[hr_column].unique()
num_unique_hr_values = len(unique_hr_values)

print(f"Numero totale di righe nel CSV (escludendo l'header): {num_total_rows}")
print(f"Numero totale di valori univoci in HR: {num_unique_hr_values}")

del df

# Conta il numero totale di righe escludendo l'header
num_total_rows = len(df)

# Estrai la colonna HR (assicurati che il nome della colonna sia corretto)
hr_column = " ECG HR"  # Sostituisci con il nome effettivo della colonna HR

# Crea un DataFrame contenente solo la colonna HR
hr_df = df[[hr_column]]

# Conta i valori univoci nella colonna HR
unique_hr_values = hr_df[hr_column].unique()
num_unique_hr_values = len(unique_hr_values)

print(f"Numero totale di righe nel CSV (escludendo l'header): {num_total_rows}")
print(f"Numero totale di valori univoci in HR: {num_unique_hr_values}")
# Rimuovi le righe in cui i valori della colonna "ECG HR" sono minori di 0
df = df[df[hr_column] >= 0]

# Ora puoi stampare di nuovo il numero totale di righe nel DataFrame
num_rows_after_removal = len(df)
print(f"Numero totale di righe nel DataFrame dopo la rimozione: {num_rows_after_removal}")

# Crea un DataFrame contenente solo la colonna HR
hr_df = df[[hr_column]]

# Rimuovi i valori negativi dalla colonna HR nel DataFrame hr_df
hr_df = hr_df[hr_df[hr_column] >= 0]

# Stampa il numero di valori univoci nella colonna HR dopo la rimozione dei valori negativi
unique_hr_values_after_removal = hr_df[hr_column].unique()
num_unique_hr_values_after_removal = len(unique_hr_values_after_removal)

print(f"Numero totale di valori univoci in HR dopo la rimozione dei valori negativi: {num_unique_hr_values_after_removal}")

# Salva il DataFrame modificato in un nuovo file CSV
new_csv_file = "/content/drive/MyDrive/Thesis <BELLIZZI>/ecg-fitness_raw-v1.0/dataset_cleaned.csv"
df.to_csv(new_csv_file, index=False)

# Puoi anche stampare il percorso del nuovo file CSV
print(f"DataFrame salvato in: {new_csv_file}")

for root, dirs, files in os.walk(main_directory):
    for file in files:
        if file.startswith("fin") and file.endswith(".csv"):
            file_path = os.path.join(root, file)

            # Carica il file CSV
            df = pd.read_csv(file_path)

            # Rimuovi le righe in cui i valori della colonna "ECG HR" sono minori di 0
            hr_column = " ECG HR"  # Sostituisci con il nome effettivo della colonna HR
            df = df[df[hr_column] >= 0]

            # Salva il DataFrame modificato nello stesso file CSV
            df.to_csv(file_path, index=False)

            print(f"File {file} aggiornato con successo.")

































