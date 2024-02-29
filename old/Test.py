import torch
from old import Transformer
import cv2
import dlib
import numpy as np
import tqdm
from old.Transformer import denormalize


model = Transformer.ViT()
model.load_state_dict(torch.load(r'F:\Università\Thesis\VIT.py'))
model.eval()

video_path = r'website/static/videos/film.mp4'
face_detector = r'F:\Università\Thesis\mmod_human_face_detector.dat'
landmark_predictor = r'F:\Università\Thesis\shape_predictor_68_face_landmarks_GTX.dat'
file_path = r'F:\Università\Thesis\min_max_values.txt'

face_detector = dlib.cnn_face_detection_model_v1(face_detector)
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

Pl = 4
Fps = 30
Fl = 0.75
Fh = 4


def draw_annotations(frame, face, predicted_value):
    x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    text = f'Predicted Value: {predicted_value:.2f}'
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

def process_and_visualize_video(video_path, face_detector, landmark_predictor, tracker, Pl, Fps, Fl, Fh, max_time_to_analyze_seconds, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] - W: Error opening video file: {video_path}")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames_to_analyze = int(max_time_to_analyze_seconds * frame_rate)
    progress_bar = tqdm(total=min(total_frames, max_frames_to_analyze), position=0, leave=True,
                        desc=f'Processing Frames for {video_path}')
    output_video_path = r'F:\Università\Thesis\static\videos\output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))
    rois_list = []
    video_images = []

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= max_frames_to_analyze:
            break

        if frame_count % 60 == 0:
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
        else:
            print("[!] - W: Issues with face region in the current frame. Skipping...")
            frame_count += 1
            continue

        progress_bar.update(1)
        frame_count += 1

    progress_bar.close()

    features_img = extract_features(rois_list, Pl, Fps, Fl, Fh)
    video_images.extend(features_img)

    for image in video_images:
        with torch.no_grad():
            output = model(image)
        predicted_value = output.item()
        pred_denorm = denormalize(predicted_value,max_mean_hr,min_mean_hr)
        annotated_frame = draw_annotations(frame.copy(), face, pred_denorm)
        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[+] - OK: Video analyzed and annotated. Output saved to {output_video_path}")


tracker = cv2.TrackerGOTURN_create()
process_and_visualize_video(video_path, face_detector, landmark_predictor, tracker, Pl, Fps, Fl, Fh, model)

