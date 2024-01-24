from flask import Blueprint, render_template
from flask_login import login_required, current_user
from flask import Blueprint, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from .models import db,Video

views = Blueprint('views',__name__)

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)

ALLOWED_EXT = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

#def pipeline(video_data):
    # Esegui le operazioni desiderate sulla video_data
    # Ad esempio, puoi utilizzare librerie come OpenCV per elaborare il video
    # o eseguire altre operazioni personalizzate.

    # # Esempio: Salva il video elaborato su disco
    # processed_video_path = 'static/processed_videos/film.mp4'
    # with open(processed_video_path, 'wb') as processed_file:
    #     processed_file.write(video_data)

    # Esempio: Esegui qualche operazione con OpenCV (richiede l'installazione di OpenCV)
    # import cv2
    # video_array = bytearray(video_data)
    # frame = cv2.imdecode(np.array(video_array), cv2.IMREAD_UNCHANGED)
    # # Esegui operazioni di elaborazione del frame
    # # ...
    # processed_frame = frame
    # processed_video_data = cv2.imencode('.mp4', processed_frame)[1].tobytes()

    # Puoi restituire il video elaborato o eseguire altre azioni a seconda delle tue esigenze
    #return processed_video_path  # o processed_video_data se hai eseguito un'elaborazione specifica

@views.route('/upload', methods=['POST'])
def upload():
    video_verified = False

    if 'video' not in request.files:
        flash('No video file found!', category='error')
        return redirect(url_for('views.home'))

    video = request.files['video']

    if video.filename == " ":
        flash('No video file selected!', category='error')
        return redirect(url_for('views.home'))

    if video and allowed_file(video.filename):

        filename = secure_filename(video.filename)
        upload_folder = 'static/videos'
        os.makedirs(upload_folder, exist_ok=True)
        video_path = os.path.join(upload_folder, filename)
        video.save(video_path)
        with open(video_path, 'rb') as file:
            video_data = file.read()
        new_video = Video(video_data=video_data)
        db.session.add(new_video)
        db.session.commit()

        flash('Video uploaded successfully!', category='success')
        video_verified = True

        # if video_verified:
        #     pipeline(video_data)
    else:
        flash('File in the wrong format. Only mp4 allowed!', category='error')

    return redirect(url_for('views.home'))






