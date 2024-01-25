from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager

db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)
    current_directory = path.dirname(path.abspath(__file__))
    project_directory = path.dirname(current_directory)
    shape_predictor_path = path.join(project_directory, r'shape_predictor_68_face_landmarks_GTX.dat')
    face_detector_path = path.join(project_directory, r'mmod_human_face_detector.dat')
    max_min_path = path.join(project_directory, r'min_max_values_vit.txt')

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Video, File

    with app.app_context():
        db.create_all()

        if not File.query.filter(
                db.or_(File.filename == shape_predictor_path,
                       File.filename == face_detector_path),
                       File.filename == max_min_path).first():

            with open(shape_predictor_path, 'rb') as file:
                content = file.read()
                file1 = File(filename=shape_predictor_path,
                             content=content)
                db.session.add(file1)

            with open(face_detector_path, 'rb') as file:
                content = file.read()
                file2 = File(filename=face_detector_path, content=content)
                db.session.add(file2)

            with open(max_min_path, 'rb') as file:
                content = file.read()
                file3 = File(filename=max_min_path, content=content)
                db.session.add(file3)

            db.session.commit()
        else:
            pass

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')