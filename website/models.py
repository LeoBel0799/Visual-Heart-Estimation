from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

from sqlalchemy import LargeBinary

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_data = db.Column(LargeBinary)
    creation_date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __init__(self, video_data, *args, **kwargs):
        super(Video, self).__init__(*args, **kwargs)
        self.video_data = video_data


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Video')

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    content = db.Column(db.LargeBinary, nullable=False)