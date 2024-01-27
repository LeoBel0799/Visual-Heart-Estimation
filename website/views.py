from flask import Response, Blueprint, render_template, send_file, jsonify
from flask_login import login_required, current_user
from flask import Blueprint, flash, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import base64

from .models import db,Video
from sqlalchemy import desc

views = Blueprint('views',__name__)

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)

ALLOWED_EXT = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def pipeline(video_data):
    #INSERIRE LA LOGICA PRESENTE IN Test.py
    return


@views.route('/choose_video', methods=['POST'])
def choose_video():

    if 'video' not in request.files:
        flash('No video file found!', category='error')
        return redirect(url_for('views.home'))

    video = request.files['video']

    if video.filename == " ":
        flash('No video file selected!', category='error')
        return redirect(url_for('views.home'))

    if video and allowed_file(video.filename):
        video_data = video.read()
        new_video = Video(video_data=video_data)
        db.session.add(new_video)
        db.session.commit()
        # filename = secure_filename(video.filename)
        # upload_folder = 'static/videos'
        # os.makedirs(upload_folder, exist_ok=True)
        # video_path = os.path.join(upload_folder, filename)
        # video.save(video_path)
        # with open(video_path, 'rb') as file:
        #     video_data = file.read()
        # new_video = Video(video_data=video_data)
        # db.session.add(new_video)
        # db.session.commit()
        flash('Video verified ad uploaded successfully!', category='success')

    else:
        flash('File in the wrong format. Only mp4 allowed!', category='error')

    return redirect(url_for('views.home'))


def index():
    last_video = Video.query.order_by(Video.id.desc()).first()
    return last_video.id if last_video else None


@views.route('/get_video', methods=['GET'])
def get_video():
    video = Video.query.get(index())
    if video:
        video_data = video.video_data
        pipeline(video_data)
        response = Response(video_data, content_type='video/mp4')
        return response
    else:
        flash('No video file found!', category='error')
        return redirect(url_for('views.home'))